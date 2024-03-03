import torch
from inverse_stable_diffusion_gs import InversableStableDiffusionPipeline
import argparse
from PIL import Image
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
from scipy.stats import norm
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import numpy as np
from scipy.stats import norm

def exactract_latents(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe = pipe.to(device)
    tester_prompt = ''
    # assume at the detection time, the original prompt is unknown，原论文中使用的就是空prompt
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    # 格式PIL.Image.Image
    orig_image = Image.open(args.orig_image_path)
    # 
    img_w = transform_img(orig_image).unsqueeze(0).to(torch.float16).to(device)
    # 
    image_latents = pipe.get_image_latents(img_w, sample=False)
    # 获得逆向后的噪声矩阵，这里与推理步数有关，越大会变好
    reversed_latents = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.num_inference_steps,
        )

    return reversed_latents.cpu()



def recover_exactracted_message(reversed_latents, key, nonce, l=1):
    # 初始化Cipher
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    decryptor = cipher.decryptor()
    
    # 从reversed_latents中恢复m的二进制表示
    reconstructed_m_bits = []
    for z_s_T_value in np.nditer(reversed_latents):
        # 使用norm.ppf的逆操作恢复原始的y值
        y_reconstructed = norm.cdf(z_s_T_value) * 2**l
        reconstructed_m_bits.append(int(y_reconstructed))

    # 将二进制位转换为字节
    m_reconstructed_bytes = bytes(int(''.join(str(bit) for bit in reconstructed_m_bits[i:i+8]), 2) for i in range(0, len(reconstructed_m_bits), 8))

    # 解密m以恢复扩散过程前的数据s_d
    s_d_reconstructed = decryptor.update(m_reconstructed_bytes) + decryptor.finalize()

    # 假设原始水印消息k是s_d的前32字节
    k_reconstructed = s_d_reconstructed[:32]

    # 将重构的k转换为十六进制表示
    k_reconstructed_hex = k_reconstructed.hex()
    
    # 将重构的k转换为二进制表示
    k_reconstructed_bin = ''.join(format(byte, '08b') for byte in k_reconstructed)
    
    return k_reconstructed_bin



def calculate_bit_accuracy(original_message_hex, extracted_message_bin):
    # Convert the original hex message to binary
    original_message_bin = bin(int(original_message_hex, 16))[2:].zfill(len(original_message_hex) * 4)
    # Ensure both binary strings are of the same length
    min_length = min(len(original_message_bin), len(extracted_message_bin))
    original_message_bin = original_message_bin[:min_length]
    extracted_message_bin = extracted_message_bin[:min_length]
    # Calculate bit accuracy
    matching_bits = sum(1 for x, y in zip(original_message_bin, extracted_message_bin) if x == y)
    bit_accuracy = matching_bits / min_length
    return original_message_bin,bit_accuracy

def get_result_for_one_image(args):
    # Process each image
    reversed_latents = exactract_latents(args)
    # 
    extracted_message_bin = recover_exactracted_message(reversed_latents, args.key, args.nonce, args.l)
    # 
    original_message_bin, bit_accuracy = calculate_bit_accuracy(args.original_message_hex, extracted_message_bin)
    return original_message_bin,extracted_message_bin,bit_accuracy

def process_directory(args):
    # Get all image files in the directory
    image_files = glob.glob(os.path.join(args.directory, "*.png")) + glob.glob(os.path.join(args.directory, "*.jpg"))

    with open("result.txt", "a") as result_file:
        for image_path in tqdm(image_files):
            args.orig_image_path = image_path
            try:
                # Process each image
                result=get_result_for_one_image(args)
                result_file.write(f"{os.path.basename(image_path)}: Original Message: {result[0]} , Extracted Message: {result[1]}, Bit Accuracy: {result[2]}\n")
            except Exception as e:
                result_file.write(f"Error processing {image_path}: {e}\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract watermark from a image')
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--directory', help='Directory containing images to process', default="/home/dongli911/.wan/Project/AIGC/stablediffusion/outputs/txt2img-samples/need2test")
    parser.add_argument('--orig_image_path', default="/home/dongli911/.wan/Project/AIGC/stablediffusion/outputs/txt2img-samples/samples/00381.png")
    parser.add_argument('--key_hex', required=True, help='Hexadecimal key used for encryption', default="5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7")
    # nonce_hex=14192f43863f6bad1bf54b7832697389
    parser.add_argument('--nonce_hex', default="", help='Hexadecimal nonce used for encryption, if empty will use part of the key')
    parser.add_argument('--original_message_hex', default="6c746865726f6c746865726f6c746865726f6c746865726f6c746865726f6c74", help='Hexadecimal representation of the original message for accuracy calculation')
    parser.add_argument('--num_inference_steps', default=100, type=int, help='Number of inference steps for the model')
    parser.add_argument('--l', default=1, type=int, help="The size of slide windows for m")
    args = parser.parse_args()

    # 将十六进制字符串转换为字节串
    args.key = bytes.fromhex(args.key_hex)
    if args.nonce_hex!="":
        # 使用参数的nonce_hex
        nonce = bytes.fromhex(args.nonce_hex)
    else:
        # 使用固定的nonce, 将nonce_hex转换为字节
        nonce = bytes.fromhex(args.key_hex[16:48])
    
    # 批处理
    process_directory(args)
    
    # 单次处理
    # get_result_for_one_image(args)



    # # 假设reversed_latents是从图像中提取的逆向噪声矩阵
    # reversed_latents = exactract_latents(args)

    # # 恢复原始消息
    # exactracted_message = recover_exactracted_message(reversed_latents, key, nonce, args.l)
    # print("Recovered message :\n", exactracted_message)

    # # 将十六进制字符串转换为二进制字符串表示
    # original_message = args.original_message_hex
    # binary_original_message = bin(int(original_message, 16))[2:].zfill(len(original_message) * 4)  # 每个十六进制数转换为4位二进制数

    # print("Original message:\n",binary_original_message)

    # # 确保 binary_string 和 exactracted_message 的长度相同
    # binary_original_message = binary_original_message[:len(exactracted_message)]

    # # 计算位准确率
    # matching_bits = sum(b1 == b2 for b1, b2 in zip(binary_original_message, exactracted_message))
    # accuracy = matching_bits / len(exactracted_message)

    # print("Bit accuracy: ",accuracy)
    
