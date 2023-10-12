
import pytest
import requests
import base64
from datetime import datetime
import json
import argparse


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--config_file', default='test/config.json')           # positional argument
parser.add_argument('--model_name', default=None)

args = parser.parse_args()

simple_txt2img_request = json.load(open(args.config_file, 'r'))


@pytest.fixture()
def url_txt2img(base_url):
    return f"{base_url}/sdapi/v1/txt2img"


def save_image_with_info(response, simple_txt2img_request, save_path="outputs/txt2img-images/{}/{}_{}.{}"):
    response = response.json()
    images = response['images']
    info = (response['info'])

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    date = datetime.now().strftime('%Y-%m-%d')
    for idx, image in enumerate(images):
        image_name = save_path.format(date, timestamp, idx, 'png')
        with open(image_name, "wb") as fh:
            fh.write(base64.decodebytes(bytes(image, encoding='utf-8')))

        image_info_name = save_path.format(date, timestamp, idx, 'json')
        with open(image_info_name, "w") as outfile:
            outfile.write(json.dumps(response, indent=4))

        config_dump_file = save_path.format(date, timestamp, idx, 'config.json')
        with open(config_dump_file, "w") as outfile:
            outfile.write(json.dumps(simple_txt2img_request, indent=4))

def test_txt2img_simple_performed(url_txt2img, simple_txt2img_request):
    response = requests.post(url_txt2img, json=simple_txt2img_request)
    assert response.status_code == 200
    save_image_with_info(response, simple_txt2img_request)


def test_txt2img_with_negative_prompt_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["negative_prompt"] = "example negative prompt"
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200


def test_txt2img_with_complex_prompt_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["prompt"] = "((emphasis)), (emphasis1:1.1), [to:1], [from::2], [from:to:0.3], [alt|alt1]"
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200


def test_txt2img_not_square_image_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["height"] = 128
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200


def test_txt2img_with_hrfix_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["enable_hr"] = True
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200


def test_txt2img_with_tiling_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["tiling"] = True
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200


def test_txt2img_with_restore_faces_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["restore_faces"] = True
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200


@pytest.mark.parametrize("sampler", ["PLMS", "DDIM", "UniPC"])
def test_txt2img_with_vanilla_sampler_performed(url_txt2img, simple_txt2img_request, sampler):
    simple_txt2img_request["sampler_index"] = sampler
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200


def test_txt2img_multiple_batches_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["n_iter"] = 2
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200


def test_txt2img_batch_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["batch_size"] = 2
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200





if __name__ == '__main__':
    test_txt2img_simple_performed("http://127.0.0.1:7861/sdapi/v1/txt2img", simple_txt2img_request)
