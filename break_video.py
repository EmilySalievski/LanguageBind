from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from moviepy import VideoFileClip, concatenate_videoclips
import os 
from pathlib import Path
from PIL import Image 
import tqdm 
import torch

# Process the video
prefix_vid = 'mini_video_'
prefix_img = 'frame_'

def process_video(video_path, fps_required):
    clip = VideoFileClip(video_path)
    fps_clip = clip.fps
    fps_required = min(fps_required, fps_clip)
    output_dir = os.path.join(str(Path.home()), 'video_outputs', Path(video_path).name.split('.')[0], f"fps_{fps_required}")
    #output_dir = os.path.join(Path(video_path).parent.absolute(), Path(video_path).name.split('.')[0], f"fps_{fps_required}")
    output_dir_frames = os.path.join(output_dir, 'frames')
    output_dir_mini_videos = os.path.join(output_dir, 'mini_videos')

    mini_video_files = []
    images = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir_frames)
        os.makedirs(output_dir_mini_videos)

    try:
        with open (f'{output_dir}/vid_details.txt', 'r') as file:
            file_contents = file.read()
            files = file_contents.split('\n')
            for file in files:
                if 'video' in file.split(',')[1]:
                    vid_max_num = int(file.split(',')[1].split('.')[0][len(prefix_vid)+1:])
                    mini_video_files = [os.path.join(output_dir_mini_videos, f"mini_video_{i}.mp4") for i in range(vid_max_num+1)]
                if 'frame' in file.split(',')[1]:
                    img_max_num = int(file.split(',')[1].split('.')[0][len(prefix_img)+1:])
                    images = [os.path.join(output_dir_frames, f"frame_{i}.jpg") for i in range(img_max_num+1)]
            return mini_video_files, images, output_dir
    except:
        print(f"Pre-processing the video segments")
        num_secs = 8//fps_required 
        # Iterate over the video in 8-second intervals
        if num_secs == 0: # when you have floats
            num_secs = num_secs = 8/fps_required
            start_time = 0
            end_time = 0 + num_secs
            keep_adding = True
            i = 0
            while keep_adding:
                mini_video = clip.subclip(start_time, end_time)
                mini_video = mini_video.with_fps(fps_required)
                file_name = f"{output_dir_mini_videos}/mini_video_{i}.mp4"
                mini_video.write_videofile(file_name, fps=fps_required)
                mini_video_files.append(file_name)

                start_time = end_time
                if start_time >=  clip.duration:
                    keep_adding = False
                else:
                    end_time =  min(start_time + num_secs, clip.duration)
                i += 1

        else:
            for i in range(0, int(clip.duration), num_secs):
                start_time = i
                end_time = min(i + num_secs, clip.duration)

                mini_video = clip.subclipped(start_time, end_time)

                if end_time == clip.duration:
                    mini_video = mini_video.with_duration(end_time - start_time).with_fps(fps_required)
                else:
                    mini_video = mini_video.with_duration(num_secs).with_fps(fps_required)

                # Define the file name for the mini video
                file_name = f"{output_dir_mini_videos}/mini_video_{i//num_secs}.mp4"

                # Write the mini video to a file
                mini_video.write_videofile(file_name, fps=fps_required)

                # Append the file name to the list
                mini_video_files.append(file_name)
        

        # print(f"Pre-processing the image frames")
        frames = clip.iter_frames(fps=fps_required)
        for i, f in tqdm.tqdm(enumerate(frames)):
            file_name = f"{output_dir_frames}/frame_{i}.jpg"
            Image.fromarray(f).save(file_name)
            images.append(file_name)
        
        print(f"I have {len(mini_video_files)} video segments.")
        print(f"I have {len(images)} frames.")

        with open (f'{output_dir}/vid_details.txt', 'w') as file:
            file.write(f'{output_dir_mini_videos}, mini_video_{len(mini_video_files)-1}.mp4\n{output_dir_frames}, frame_{len(images)-1}.jpg')

        return mini_video_files, images, output_dir

video_path = '/home/datasets/ego4d_data/v2/full_scale/fffbaeef-577f-45f0-baa9-f10cabf62dfb.mp4'
mini_video_files, images, output_dir = process_video(video_path, fps_required=8)

video = mini_video_files
image = images

# rest of langugae bind
device = 'cuda:0'
device = torch.device(device)
clip_type = {
    'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
}

model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
model = model.to(device)
model.eval()
pretrained_ckpt = f'lb203/LanguageBind_Image'
tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}


language = ["Person waving", "Dog running in a field"]
inputs = {
    'video': to_device(modality_transform['video'](video), device),
}
inputs['language'] = to_device(tokenizer(language, max_length=77, padding='max_length',
                                            truncation=True, return_tensors='pt'), device)

with torch.no_grad():
    embeddings = model(inputs)

print("Video x Text: \n",
        torch.softmax(embeddings['video'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
