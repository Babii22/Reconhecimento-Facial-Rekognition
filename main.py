import boto3
import json
import cv2
import numpy as np
from pathlib import Path


rekognition = boto3.client("rekognition")

image_paths = [
    r"C:\Python DIO\Reconhecimento-Facial\Imagens\pink-taylor.png",
    r"C:\Python DIO\Reconhecimento-Facial\Imagens\shakira-fans.png",
    r"C:\Python DIO\Reconhecimento-Facial\Imagens\katy-shakira.png"
]

target_singers = {"Katy Perry", "Shakira", "Taylor Swift", "Pink"}

def detect_celebrities(image_path):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    response = rekognition.recognize_celebrities(Image={"Bytes": image_bytes})
    return response

def draw_bounding_boxes(image_path, response):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return None

    height, width, _ = image.shape

    for celebrity in response["CelebrityFaces"]:
        name = celebrity["Name"]
        if name in target_singers:
            box = celebrity["Face"]["BoundingBox"]
            x1 = int(box["Left"] * width)
            y1 = int(box["Top"] * height)
            x2 = int((box["Left"] + box["Width"]) * width)
            y2 = int((box["Top"] + box["Height"]) * height)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 200, 150), 1)
            cv2.putText(image, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 150), 1)

    return image

def process_images():
    for image_path in image_paths:
        print(f"Processando {image_path}...")
        
        if not Path(image_path).exists():
            print(f"Erro: Arquivo {image_path} não encontrado!\n")
            continue

        response = detect_celebrities(image_path)

        output_json = str(Path(image_path).with_stem(Path(image_path).stem + "_output").with_suffix(".json"))
        output_image = str(Path(image_path).with_stem(Path(image_path).stem + "_marked").with_suffix(".png"))

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=4)
        print(f"Resultados salvos em {output_json}")

        processed_image = draw_bounding_boxes(image_path, response)
        if processed_image is not None:
            cv2.imwrite(output_image, processed_image)
            print(f"Imagem processada salva como {output_image}\n")

if __name__ == "__main__":
    process_images()