from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, eval_model_many
from PIL import Image
from datasets import load_dataset

def resize_image(image, max_size=1800):
    # Open the image
    # image = Image.open(input_path)

    # Get the original width and height
    original_width, original_height = image.size

    # Calculate the new size while maintaining the aspect ratio
    if original_width > original_height:
        new_width = max_size
        new_height = int(original_height * (max_size / original_width))
    else:
        new_height = max_size
        new_width = int(original_width * (max_size / original_height))

    return image.resize((new_width, new_height), Image.LANCZOS)


prompt_2 = """
I like you to pretend to be an architect rendering professional that have to generate captions for images following the next guidelines

Guideline:
The captions must describe the visible elements in the image, with particular emphasis on the main buildings. In this way, you should describe the type, elements, and materials present in the building in great detail. The other elements in the image don't need as much detail but must be present in the captions. 

Please, maintain a regular structure for the captions. The order of priority for describing the scene is as follows:

<building type>, <material of the walls and color>, <other elements of the building>, <angle view>, <interior/exterior>, <time of the day>, <objects present in the scene>, <wheater>, <lightinig effects, blurry etc.>

Building type. Describe the type of the building e.g. residential apartments, industrial facility, museum, Hotel…
Materials and color. Prioritize the main material present in the building (usually its walls). Describe also its colors e.g. “grey polished concrete”, “burgundy bricks”...

Interior/exterior: whether the scene is interior or exterior setting.
Time of the day. Can be just one word or a time of day with a short description if you think it helps to describe the lighting of the scene: “sunset”, “midday”, “night moonlight”, “summer bright day”, “dusk golden hour”...
Wheater: other elements of the scene such as “fog”, “snow”, “rain”, or any other element you find to contribute to how the scene looks.
Other objects present in the scene: Describe other elements than the main buildings, like trees, cars, furniture, people, animals, other construction structures… You can also use verbs and adjectives e.g. “man fishing” or “couple sited in a table”.
Lightning effects, blurry, etc. Can include elements related to the camera or rendering effects, such as “blurred foreground”, “lens flare”, “overexposed sky”, “long exposition”, “backlighting”... Or other effects from the environment: “reflection on water”, “ripple effect”...
"""

image_file = "arch-test-11.jpg"

# model_path = "liuhaotian/llava-v1.5-7b"
model_path = "liuhaotian/llava-v1.5-13b"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt_2,
    "conv_mode": None,
    "image_file": "arch-test-11.jpg",
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path))

dataset = load_dataset("danabib/archdaily_captioned_elsed", split="train")

# Download
download_folder = "archdaily-llava-10k"
num_images = len(dataset)

for i in range(num_images):
    if dataset[i]["image"].width > 1800 or dataset[i]["image"].height > 1800:
        args.image_file = resize_image(dataset[i]["image"])
    else:
        args.image_file = dataset[i]["image"]
        
    try:
        args.image_file = dataset[i]["image"]
        caption = eval_model_many(args, model_name, tokenizer, model, image_processor, context_len)
        # print(f"Downloading Image {i}/{num_images}")
        dataset[i]["image"].save(f"{download_folder}/archviz_llava_{i:07d}.png", "PNG")
        print(f"Caption: {caption}")
        with open(f"{download_folder}/archviz_llava_{i:07d}.txt", "w") as file:
            print("inside with")
            file.write(caption) 
    except Exception as e:
        # Log the error to a file
        with open("bad_images_llava.txt", "a") as file:
            file.write(f"Error with image {i}: {e}\n")
        print(e)