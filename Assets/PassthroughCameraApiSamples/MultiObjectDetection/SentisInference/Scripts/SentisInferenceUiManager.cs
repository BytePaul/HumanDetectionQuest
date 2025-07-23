// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections.Generic;
using Meta.XR.Samples;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class SentisInferenceUiManager : MonoBehaviour
    {
        [Header("Placement configureation")]
        [SerializeField] private EnvironmentRayCastSampleManager m_environmentRaycast;
        [SerializeField] private WebCamTextureManager m_webCamTextureManager;
        private PassthroughCameraEye CameraEye => m_webCamTextureManager.Eye;

        [Header("UI display references")]
        [SerializeField] private SentisObjectDetectedUiManager m_detectionCanvas;
        [SerializeField] private RawImage m_displayImage;
        [SerializeField] private Sprite m_boxTexture;
        [SerializeField] private Color m_boxColor;
        [SerializeField] private Font m_font;
        [SerializeField] private Color m_fontColor;
        [SerializeField] private int m_fontSize = 80;
        [Space(10)]
        public UnityEvent<int> OnObjectsDetected;

        [Header("Human Detection Configuration")]
        [Tooltip("The class name string for 'person' as it appears in your model's labels.txt. " +
                 "Ensure this matches the casing in your labels file (e.g., 'person', 'Person', 'human').")]
        [SerializeField] private string m_personClassName = "person"; // Default to "person" for COCO models

        public List<BoundingBox> BoxDrawn = new();

        private string[] m_labels;
        private List<GameObject> m_boxPool = new();
        private Transform m_displayLocation;

        //bounding box data
        public struct BoundingBox
        {
            public float CenterX;
            public float CenterY;
            public float Width;
            public float Height;
            public string Label;
            public Vector3? WorldPos;
            public string ClassName;
        }

        #region Unity Functions
        private void Start()
        {
            m_displayLocation = m_displayImage.transform;
        }
        #endregion

        #region Detection Functions
        public void OnObjectDetectionError()
        {
            // Clear current boxes
            ClearAnnotations();

            // Set object found to 0
            OnObjectsDetected?.Invoke(0);
        }
        #endregion

        #region BoundingBoxes functions
        public void SetLabels(TextAsset labelsAsset)
        {
            //Parse neural net m_labels
            m_labels = labelsAsset.text.Split('\n');
            Debug.Log($"[SentisUI] Labels loaded. Total labels: {m_labels.Length}");
            for (int i = 0; i < m_labels.Length; i++)
            {
                Debug.Log($"[SentisUI] Label[{i}]: '{m_labels[i].Trim()}'"); // .Trim() to show exact string without whitespace
            }
        }

        public void SetDetectionCapture(Texture image)
        {
            m_displayImage.texture = image;
            m_detectionCanvas.CapturePosition();
        }

        public void DrawUIBoxes(Tensor<float> output, Tensor<int> labelIDs, float imageWidth, float imageHeight)
        {
            // Update canvas position
            m_detectionCanvas.UpdatePosition();

            // Clear current boxes
            ClearAnnotations();

            var displayWidth = m_displayImage.rectTransform.rect.width;
            var displayHeight = m_displayImage.rectTransform.rect.height;

            var scaleX = displayWidth / imageWidth;
            var scaleY = displayHeight / imageHeight;

            var halfWidth = displayWidth / 2;
            var halfHeight = displayHeight / 2;

            var boxesFoundInOutput = output.shape[0]; // Total boxes from Sentis output, before filtering
            Debug.Log($"[SentisUI] Raw boxes found in model output: {boxesFoundInOutput}");

            if (boxesFoundInOutput <= 0)
            {
                OnObjectsDetected?.Invoke(0);
                return;
            }
            var maxBoxesToProcess = Mathf.Min(boxesFoundInOutput, 200);

            int humanDetectionsCount = 0; // Counter for "person" detections

            //Get the camera intrinsics
            var intrinsics = PassthroughCameraUtils.GetCameraIntrinsics(CameraEye);
            var camRes = intrinsics.Resolution;

            //Draw the bounding boxes
            for (var n = 0; n < maxBoxesToProcess; n++)
            {
                // Get object class name from the labels array
                // IMPORTANT: Adding .Trim() here to remove any leading/trailing whitespace from the label string
                var rawClassnameFromLabels = m_labels[labelIDs[n]];
                var processedClassname = rawClassnameFromLabels.Trim().Replace(" ", "_"); // Added .Trim()

                // --- DEBUG LOGS FOR FILTERING ---
                Debug.Log($"[SentisUI] --- Processing Detection {n} ---");
                Debug.Log($"[SentisUI] Label ID from model: {labelIDs[n]}");
                Debug.Log($"[SentisUI] Raw classname from labels[{labelIDs[n]}]: '{rawClassnameFromLabels}'");
                Debug.Log($"[SentisUI] Processed classname (trimmed/replaced): '{processedClassname}'");
                Debug.Log($"[SentisUI] Configured m_personClassName: '{m_personClassName}'");
                // Adding .Trim() to m_personClassName for the comparison as well, to be consistent.
                Debug.Log($"[SentisUI] Comparison: '{processedClassname.ToLowerInvariant()}' == '{m_personClassName.ToLowerInvariant().Trim()}' ? {processedClassname.ToLowerInvariant() == m_personClassName.ToLowerInvariant().Trim()}");
                // --- END DEBUG LOGS ---


                // --- NEW FILTERING LOGIC ---
                // Only process and draw if the detected class is "person"
                if (processedClassname.ToLowerInvariant() != m_personClassName.ToLowerInvariant().Trim()) // Added .Trim() to m_personClassName here
                {
                    Debug.Log($"[SentisUI] Skipping non-person detection: '{processedClassname}'");
                    continue; // Skip this detection if it's not a person
                }
                // --- END NEW FILTERING LOGIC ---

                Debug.Log($"[SentisUI] === PERSON DETECTED! Processing to draw box. ===");
                humanDetectionsCount++;

                // Get bounding box center coordinates
                var centerX = output[n, 0] * scaleX - halfWidth;
                var centerY = output[n, 1] * scaleY - halfHeight;
                var perX = (centerX + halfWidth) / displayWidth;
                var perY = (centerY + halfHeight) / displayHeight;

                // Get the 3D marker world position using Depth Raycast
                var centerPixel = new Vector2Int(Mathf.RoundToInt(perX * camRes.x), Mathf.RoundToInt((1.0f - perY) * camRes.y));
                var ray = PassthroughCameraUtils.ScreenPointToRayInWorld(CameraEye, centerPixel);
                var worldPos = m_environmentRaycast.PlaceGameObjectByScreenPos(ray);

                // Create a new bounding box
                var box = new BoundingBox
                {
                    CenterX = centerX,
                    CenterY = centerY,
                    ClassName = processedClassname, // Use processed classname here
                    Width = output[n, 2] * scaleX,
                    Height = output[n, 3] * scaleY,
                    Label = $"Id: {n} Class: {processedClassname} Center (px): {(int)centerX},{(int)centerY} Center (%): {perX:0.00},{perY:0.00}",
                    WorldPos = worldPos,
                };

                // Add to the list of boxes
                BoxDrawn.Add(box);

                // Draw 2D box
                DrawBox(box, humanDetectionsCount - 1); // Adjust ID for pooling based on filtered count
            }

            // Update the event with the number of *filtered* objects
            Debug.Log($"[SentisUI] Total human detections drawn this frame: {humanDetectionsCount}");
            OnObjectsDetected?.Invoke(humanDetectionsCount);
        }

        private void ClearAnnotations()
        {
            foreach (var box in m_boxPool)
            {
                box?.SetActive(false);
            }
            m_boxPool.Clear(); // Clear the pool entirely as we are re-populating it with only human detections
            BoxDrawn.Clear();
            Debug.Log("[SentisUI] Annotations cleared.");
        }

        private void DrawBox(BoundingBox box, int id)
        {
            //Create the bounding box graphic or get from pool
            GameObject panel;
            // The pooling logic needs to be slightly adjusted if you want to reuse panels
            // based on a filtered list. For simplicity and to ensure correct drawing,
            // we'll create new ones for each *filtered* detection or reuse if available.
            // A more robust pooling would typically involve a separate pool manager.
            if (id < m_boxPool.Count)
            {
                panel = m_boxPool[id];
                if (panel == null)
                {
                    panel = CreateNewBox(m_boxColor);
                    m_boxPool[id] = panel; // Update pool reference
                }
                else
                {
                    panel.SetActive(true);
                }
            }
            else
            {
                panel = CreateNewBox(m_boxColor);
            }

            //Set box position
            panel.transform.localPosition = new Vector3(box.CenterX, -box.CenterY, 0.0f); // WorldPos is used for 3D marker, not UI position

            //Set box size
            var rt = panel.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(box.Width, box.Height);
            //Set label text
            var label = panel.GetComponentInChildren<Text>();
            label.text = box.Label;
            label.fontSize = 12; // Adjusted to a more common font size for UI readability
        }

        private GameObject CreateNewBox(Color color)
        {
            //Create the box and set image
            var panel = new GameObject("ObjectBox");
            _ = panel.AddComponent<CanvasRenderer>();
            var img = panel.AddComponent<Image>();
            img.color = color;
            img.sprite = m_boxTexture;
            img.type = Image.Type.Sliced;
            img.fillCenter = false;
            panel.transform.SetParent(m_displayLocation, false);

            //Create the label
            var text = new GameObject("ObjectLabel");
            _ = text.AddComponent<CanvasRenderer>();
            text.transform.SetParent(panel.transform, false);
            var txt = text.AddComponent<Text>();
            txt.font = m_font;
            txt.color = m_fontColor;
            txt.fontSize = m_fontSize;
            txt.horizontalOverflow = HorizontalWrapMode.Overflow;

            var rt2 = text.GetComponent<RectTransform>();
            rt2.offsetMin = new Vector2(20, rt2.offsetMin.y);
            rt2.offsetMax = new Vector2(0, rt2.offsetMax.y);
            rt2.offsetMin = new Vector2(rt2.offsetMin.x, 0);
            rt2.offsetMax = new Vector2(rt2.offsetMax.x, 30);
            rt2.anchorMin = new Vector2(0, 0);
            rt2.anchorMax = new Vector2(1, 1);

            m_boxPool.Add(panel); // Add to pool immediately upon creation
            return panel;
        }
        #endregion
    }
}