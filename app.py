import io
import torchvision
from flask import Flask, request, jsonify ,render_template,send_from_directory
from time import time
import os
from gtts import gTTS
import torch
import argparse
import pickle
from argparse import Namespace
from PIL import Image
from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from flask_cors import CORS 
from utils.language_utils import tokens2description
import uuid
from datetime import datetime


# Define the Flask app
app = Flask(__name__, template_folder="templates")
CORS(app)

img_size = 384

def preprocess_image(image_data, img_size):
    transf_1 = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size))])
    transf_2 = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

    pil_image = Image.open(io.BytesIO(image_data))
    if pil_image.mode != 'RGB':
        pil_image = Image.new("RGB", pil_image.size)
    preprocess_pil_image = transf_1(pil_image)
    image = torchvision.transforms.ToTensor()(preprocess_pil_image)
    image = transf_2(image)
    return image.unsqueeze(0)

#captured_image_data = request.form.get("captured_image")
#uploaded_image = request.files['image']
# Define the route for generating image captions
@app.route('/api/caption', methods=['POST'])
def generate_caption_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'})
    start = time()
    image_data = request.files['image'].read()
    image = preprocess_image(image_data, img_size)

    beam_search_kwargs = {'beam_size': 5,  # You can set your desired beam size
                          'beam_max_seq_len': 74,  # Adjust if needed
                          'sample_or_max': 'max',
                          'how_many_outputs': 1,
                          'sos_idx': sos_idx,
                          'eos_idx': eos_idx}

    with torch.no_grad():
        pred, _ = model(enc_x=image, enc_x_num_pads=[0], mode='beam_search', **beam_search_kwargs)

    pred_description = tokens2description(pred[0][0], coco_tokens['idx2word_list'], sos_idx, eos_idx)
    # print('\n\tDescription: ' + pred + '\n')
    stop = time()
    print('Time: {:.4f}s\n'.format(stop-start))
    # Save audio
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    audioname=f'caption_{timestamp}_{unique_id}.mp3'
    audio_path = "Audio/"+audioname
    myobj = gTTS(text=pred_description, lang='en', slow=False)
    myobj.save(audio_path)

    return jsonify({'caption': pred_description, 'audio_path': audioname})

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('Audio', filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--max_seq_len', type=int, default=74)
    parser.add_argument('--load_path', type=str, default='./rf_model.pth')
    parser.add_argument('--image_paths', type=str,
                        default=['./demo_material/tatin.jpg',
                                 './demo_material/micheal.jpg',
                                 './demo_material/napoleon.jpg',
                                 './demo_material/cat_girl.jpg'],
                        nargs='+')
    parser.add_argument('--beam_size', type=int, default=5)

    args = parser.parse_args()

    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)
    model_args = Namespace(model_dim=args.model_dim,
                           N_enc=args.N_enc,
                           N_dec=args.N_dec,
                           dropout=0.0,
                           drop_args=drop_args)

    with open('./demo_material/demo_coco_tokens.pickle', 'rb') as f:
        coco_tokens = pickle.load(f)
        sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
        eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

    print("Dictionary loaded ...")

    
    model = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                                swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                                swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                                swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                                swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                                swin_use_checkpoint=False,
                                final_swin_dim=1536,

                                d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                num_exp_enc_list=[32, 64, 128, 256, 512],
                                num_exp_dec=16,
                                output_word2idx=coco_tokens['word2idx_dict'],
                                output_idx2word=coco_tokens['idx2word_list'],
                                max_seq_len=args.max_seq_len, drop_args=model_args.drop_args,
                                rank='cpu')
    checkpoint = torch.load(args.load_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded ...")
    app.run(debug=True)
