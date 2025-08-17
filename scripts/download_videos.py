import youtube_dl

def download_video(youtube_url, output_path):
    """Download video from YouTube URL"""
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path,
        'quiet': True
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

if __name__ == '__main__':
    # Example badminton match video
    url = 'https://www.youtube.com/shorts/xw9_0NdwdDc'
    output = '../data/sample_videos/sample_match.mp4'
    download_video(url, output)