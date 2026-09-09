# YOLO11-PD

**YOLO11-PD: A Wavelet Transform and Depthwise Separable Convolution-Based Method for Pavement Distress Detection**

YOLO11-PD is a lightweight pavement distress detection model based on YOLO11n, designed for potholes, transverse/longitudinal cracks, and alligator cracks.

### Highlights
- **FMBFPN + PDCB** for multi-scale pothole feature enhancement.
- **WTConv + DR-Detect** for slender crack feature extraction and localization.
- **SCSP** for improved alligator crack recognition.

### Results
On N-RDD2024, YOLO11-PD reduces parameters by **6.6%** and GFLOPs by **1.6%**, while improving **mAP50 by 2.7%** and **mAP50:95 by 2.0%** over YOLO11n.

Generalization experiments were also conducted on **UAPD** and **UAV-PDD2023**.

### License
This project is based on Ultralytics YOLO11 and follows the AGPL-3.0 License.