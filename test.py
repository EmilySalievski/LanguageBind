import torch
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
import os
import glob

if __name__ == '__main__':
    device = 'cuda:0'
    device = torch.device(device)
    clip_type = {
        'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
        'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
        'thermal': 'LanguageBind_Thermal',
        'image': 'LanguageBind_Image',
        'depth': 'LanguageBind_Depth',
    }

    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'lb203/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    # Choose video here
    video_id = "fffbaeef-577f-45f0-baa9-f10cabf62dfb"
    ident = "sal161"
    mini_videos_path = base_path = f"/home/{ident}/video_outputs/{video_id}/fps_8/mini_videos"

    video = glob.glob(os.path.join(base_path, "*.mp4"))

    #video = ['/home/sal161/video_outputs/fffbaeef-577f-45f0-baa9-f10cabf62dfb/fps_8/mini_videos/mini_video_1.mp4', '/home/sal161/video_outputs/fffbaeef-577f-45f0-baa9-f10cabf62dfb/fps_8/mini_videos/mini_video_2.mp4', '/home/sal161/video_outputs/fffbaeef-577f-45f0-baa9-f10cabf62dfb/fps_8/mini_videos/mini_video_3.mp4', '/home/sal161/video_outputs/fffbaeef-577f-45f0-baa9-f10cabf62dfb/fps_8/mini_videos/mini_video_4.mp4']
    language = [
        "Where was the Russell Stover before I picked it up?", 
        "I totally got that feeling, because just the other day I picked up a snack and then pondered its journey. It made me wonder, where was that Russell Stover before you picked it up?", 
        "Hey, I was just wondering, you know, like when I pick up my own snacks, I always think about where they've been. So, where was that Russell Stover before you picked it up, and what state was it in?"
        ]

    inputs = {
        'video': to_device(modality_transform['video'](video), device),
    }
    inputs['language'] = to_device(tokenizer(language, max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), device)

    batch_size = 2  # tune based on GPU memory
    all_video_embeddings = []

    with torch.no_grad():
        for i in range(0, len(video), batch_size):
            batch_files = video[i:i+batch_size]
            batch_tensor = to_device(modality_transform['video'](batch_files), device)

            batch_inputs = {
                'video': batch_tensor,
                'language': to_device(tokenizer(language, max_length=77, padding='max_length',
                                                truncation=True, return_tensors='pt'), device),
            }

            batch_emb = model(batch_inputs)
            all_video_embeddings.append(batch_emb['video'].cpu())  # move to CPU to free GPU

    # concatenate back to one tensor
    video_embeddings = torch.cat(all_video_embeddings, dim=0).to(device)
    language_embeddings = batch_emb['language']

    # raw similarity scores - dot product between each video and each prompt embedding. Bigger = more similarity
    v = video_embeddings @ language_embeddings.T
    #print(v)


    # print("Video x Text: \n", # relative to other prompts - each row sums to 1 and indicates which is most likely to correspond to prompt
    #      torch.softmax(embeddings['video'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())

    k = 3 # finding top 3 most similar segments for each prompt
    num_videos, num_prompts = v.shape
    k = min(k, num_videos)  # safety if k > num_videos

    # Vectorized top-k: along videos (dim=0) for each prompt/column
    vals, idxs = torch.topk(v, k=k, dim=0)  # vals, idxs have shape [k, num_prompts]

    for j in range(num_prompts):
        print(f"\nPrompt {j} ({language[j]}):")
        for r in range(k):
            vid_idx = idxs[r, j].item()
            sim = vals[r, j].item()
            print(f"  Top {r+1}: Video {vid_idx+1} ({video[vid_idx]}) → similarity={sim:.7f}")
