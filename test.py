import torch
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

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

    #image = ['assets/image/0.jpg', 'assets/image/1.jpg']
    video = ['/home/sal161/video_outputs/fffbaeef-577f-45f0-baa9-f10cabf62dfb/fps_8/mini_videos/mini_video_1.mp4', '/home/sal161/video_outputs/fffbaeef-577f-45f0-baa9-f10cabf62dfb/fps_8/mini_videos/mini_video_2.mp4', '/home/sal161/video_outputs/fffbaeef-577f-45f0-baa9-f10cabf62dfb/fps_8/mini_videos/mini_video_3.mp4', '/home/sal161/video_outputs/fffbaeef-577f-45f0-baa9-f10cabf62dfb/fps_8/mini_videos/mini_video_4.mp4']
    language = ["Person waving", "Training a parakeet to climb up a ladder.", 'A lion climbing a tree to catch a monkey.']

    inputs = {
        #'image': to_device(modality_transform['image'](image), device),
        'video': to_device(modality_transform['video'](video), device),
    }
    inputs['language'] = to_device(tokenizer(language, max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), device)

    with torch.no_grad():
        embeddings = model(inputs)

    print("Video x Text: \n",
          torch.softmax(embeddings['video'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
    #print("Image x Text: \n",
    #      torch.softmax(embeddings['image'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())

#     v = embeddings['video'] @ embeddings['language'].T
#     s = torch.softmax(v, dim=0)
#     s_flattened = s.view(-1)
#     _, indices_s = torch.topk(s_flattened, 1) # if you want to select the top5 segemtns

#     print(_, indices_s)