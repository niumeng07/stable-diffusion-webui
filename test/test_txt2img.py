
import pytest
import requests
import base64
from datetime import datetime
import json


@pytest.fixture()
def url_txt2img(base_url):
    return f"{base_url}/sdapi/v1/txt2img"


@pytest.fixture()
def simple_txt2img_request():
    return {
        "batch_size": 1,
        "cfg_scale": 7,
        "denoising_strength": 0,
        "enable_hr": False,
        "eta": 0,
        "firstphase_height": 0,
        "firstphase_width": 0,
        "height": 64,
        "n_iter": 1,
        "negative_prompt": "",
        "prompt": "example prompt",
        "restore_faces": False,
        "s_churn": 0,
        "s_noise": 1,
        "s_tmax": 0,
        "s_tmin": 0,
        "sampler_index": "Euler a",
        "seed": -1,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "steps": 3,
        "styles": [],
        "subseed": -1,
        "subseed_strength": 0,
        "tiling": False,
        "width": 64,
    }


simple_txt2img_request2 = {
        "batch_size": 1, "cfg_scale": 9, "denoising_strength": 0, "enable_hr": False, "eta": 0, "firstphase_height": 0,
        "firstphase_width": 0, "height": 1824, "n_iter": 1,
        "negative_prompt": "BadDream, (UnrealisticDream:1.3)",
        "prompt": "(masterpiece), (extremely intricate:1.3), (realistic), portrait of a girl, the most beautiful in the world, (medieval armor), metal reflections, upper body, outdoors, intense sunlight, far away castle, professional photograph of a stunning woman detailed, sharp focus, dramatic, award winning, cinematic lighting, octane render unreal engine, volumetrics dtx, (film grain, blurry background, blurry foreground, bokeh, depth of field, sunset, motion blur:1.3), chainmail",
        "restore_faces": False,
        "s_churn": 0, "s_noise": 1, "s_tmax": 0, "s_tmin": 0, "sampler_index": "DPM++ SDE Karras", "seed": 5775713, "seed_resize_from_h": -1,
        "seed_resize_from_w": -1, "steps": 30, "styles": [], "subseed": -1, "subseed_strength": 0, "tiling": False, "width": 1120,
        }


def save_image_with_info(response, save_path="outputs/txt2img-images/{}/{}_{}.{}"):
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

def test_txt2img_simple_performed(url_txt2img, simple_txt2img_request):
    response = requests.post(url_txt2img, json=simple_txt2img_request)
    assert response.status_code == 200
    save_image_with_info(response)


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






test_txt2img_simple_performed("http://127.0.0.1:7861/sdapi/v1/txt2img", simple_txt2img_request2)
