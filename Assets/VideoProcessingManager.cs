using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using System.Diagnostics;
using System.IO;
using System.Collections;
using UnityFileBrowser;
using TMPro;

public class VideoProcessingManager : MonoBehaviour
{
    public Button uploadButton;
    public VideoPlayer processedVideoPlayer;
    public TextMeshProUGUI statusText; 

    private string videoPath;
    private string processedVideoPath;
    private string targetFolder;

    public void Start()
    {
        processedVideoPlayer = GetComponent<VideoPlayer>();
        
        // Hide the status text initially
        statusText.gameObject.SetActive(false);
        // processedVideoPlayer.gameObject.SetActive(false);

        // Assign button click handler
        // uploadButton.onClick.AddListener(() => { OpenFileBrowser(); });
        // uploadButton.onClick.AddListener(OpenFileBrowser);
    }

    // Function to handle file upload and start video processing
    public void OpenFileBrowser()
    {
        // Open a file dialog to select a video file
        videoPath = OpenFileDialog();

        if (!string.IsNullOrEmpty(videoPath))
        {
            // Proceed to process the video if a valid path is selected
            UploadAndProcessVideo();
        }
        else
        {
            statusText.gameObject.SetActive(true);
            statusText.text = "No video selected.";
        }
    }

    // Function to upload and process the selected video
    public void UploadAndProcessVideo()
    {
        // Set the target folder within the project (e.g., "Assets/UploadedVideos/")
        targetFolder = Application.dataPath + "/UploadedVideos/";

        // Create the folder if it doesn't exist
        if (!Directory.Exists(targetFolder))
        {
            Directory.CreateDirectory(targetFolder);
        }

        // Hide the upload button and show "Processing..." text
        // uploadButton.gameObject.SetActive(false);
        statusText.gameObject.SetActive(true);
        statusText.text = "Processing...";

        // Run the Python script to process the video
        RunPythonScript(videoPath);
    }

    // Run the Python script with the given video path
    public void RunPythonScript(string inputVideoPath)
    {
        // Set the paths
        string pythonFilePath = Application.dataPath + "/PythonScript/main.py";
        processedVideoPath = targetFolder + Path.GetFileName(inputVideoPath);

        // Start the Python process
        ProcessStartInfo startInfo = new ProcessStartInfo();
        startInfo.FileName = "/opt/anaconda3/bin/python3";
        startInfo.Arguments = string.Format("\"{0}\" --path_input_video \"{1}\" --path_output_video \"{2}\"", pythonFilePath, inputVideoPath, processedVideoPath);
        startInfo.UseShellExecute = false;
        startInfo.RedirectStandardOutput = true;
        startInfo.RedirectStandardError = true;
        startInfo.CreateNoWindow = true;

        Process process = Process.Start(startInfo);
        process.WaitForExit();

        // Check if the processing was successful
        if (process.ExitCode == 0)
        {
            UnityEngine.Debug.Log("Video processed successfully.");
            processedVideoPlayer.gameObject.SetActive(true);
            StartCoroutine(PlayProcessedVideo());  // Start playing the video after processing
        }
        else
        {
            statusText.text = "Processing failed.";
            UnityEngine.Debug.LogError("Python script failed: " + process.StandardError.ReadToEnd());
        }
    }

    // Coroutine to handle video playback
    IEnumerator PlayProcessedVideo()
    {
        if (File.Exists(processedVideoPath))
        {
            UnityEngine.Debug.Log("Processed video file exists at: " + processedVideoPath);

            // Set up and play the video using a coroutine for safety
            processedVideoPlayer.Stop();
            processedVideoPlayer.url = processedVideoPath;
            processedVideoPlayer.Prepare();

            // Wait until the video is prepared
            yield return new WaitUntil(() => processedVideoPlayer.isPrepared);

            // Play the video once it's ready
            processedVideoPlayer.Play();
            statusText.text = "Processing Complete!";

            // Show the upload button after processing
            // uploadButton.gameObject.SetActive(true);
        }
        else
        {
            statusText.text = "Failed to find processed video.";
            UnityEngine.Debug.LogError("Processed video file not found at: " + processedVideoPath);
        }
    }

    // Function to open a file dialog and return the selected file path
    public string OpenFileDialog()
    {
        var paths = FileBrowser.OpenFileBrowser(new[] { "mp4", "MP4", "avi", "mov" });
        UnityEngine.Debug.Log("Files selected: " + paths[0]);
        return paths[0];
    }
}
