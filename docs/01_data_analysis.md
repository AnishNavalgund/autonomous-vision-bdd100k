# BDD100K Dataset Analysis

This document summarizes the exploratory data analysis (EDA) performed on the BDD100K dataset for the object detection task. The goal of this analysis is to understand dataset characteristics, identify potential challenges, and predict how these factors might influence model performance.

---

## 1. Dataset Overview

### Basic Statistics

- Total Images: 79,863
  - Training: 69,863 (87.5%)
  - Validation: 10,000 (12.5%)
- Total Annotations: 1,472,397
  - Training: 1,286,871 (87.4%)
  - Validation: 185,526 (12.6%)
- Image Resolution: 1280 × 720 pixels
- Classes: 10 object detection categories

Note: 137 images annotations are missing and hence not included in the analysis.

## 2. In-Depth Analysis

### 1. Class Distribution

The dataset is highly imbalanced, with cars dominating (>55% of labels) and some classes like train almost 0. This will cause the model to perform very well on frequent classes but poorly on rare ones, leading to low recall and AP for underrepresented categories.

---

### 2. Objects per Image

So distribution of objects per image in both trian and val follow similar structure. So, the split is good and balanced in terms of objects per image.
Most images contain about 18 objects/image with a few crowded scenes even having 50+ objects. This also tells that, there are more occluded objects in the scene. So, the model must handle overlapping boxes, increasing the risk of missed detections

---

### 3. Occlusion and Truncation

Both train and val sets have similar occlusion/truncation patterns, so no data split bias. As discussed earlier, due to high oclusion, model might face partial visibility and this might increase False Negative. eg. a car might be occluded by another car, so the model might miss it. Moderate truncation especially for train, bus, truck is expected as they are longer objects and cannot be completely captured in the image.

---

### 4. Bounding Box Area Distribution

Good balance between train and val sets. This aligns with earlier findings like high object density (18 objects/image) and frequent occlusion mean many objects are either far from the camera or partially visible, resulting in small bounding boxes. For the model, this suggests precision for small objects will be challenging - for classes like traffic lights/signs, bikes, and persons. Larger objects (buses, trucks, close-up cars) will be easier to detect.

---

### 5. Weather, Scene, and Time of Day Distributions

#### Weather
Both train and val sets are dominated by clear-weather images which indicates that, cars are driven mostly in clear conditions. Foggy weather is extremely rare. 


#### Scene
The dataset is heavily biased toward city street scenes, followed by highways. This reflects typical urban driving environments where the data was collected, with higher object density and more diverse traffic elements likely in city streets compared to other scenes like parking lots or tunnels.

#### Time of Day
Daytime images dominate, and then night scenes. This reflects real-world driving habits, where most trips happen during the day and fewer during dawn or dusk because of the time duration. 

This domain imbalance could lead to reduced recall and higher misclassification rates in environments where there is less data points.

---

### 5. Class Co-occurrence

Car has the highest co-occurrence with almost every other class, which makes sense in road scenes where cars are the dominant object type. As stater earlier that, most of the images are from city streets, hence we see that car frequently appears with traffic lights and traffic signs also. Person also often co-occurs with cars, trucks, and traffic control objects, indicating pedestrian presence near roads. Truck–car and bus–car combinations are common in road scenes. Train has very low co-occurrence with all classes, which is expected as it is less rare to see trains in the city streets.

For model, car–traffic sign, car–traffic light will be easier to detect together because the model will repeatedly see them together. Rare combinations like train–X will be harder for the model to learn

---

### 7. Class Frequency by Time of Day / Weather / Scene

Class vs Scene/Weather/Time of Day confirms the analysis done earlier. The dataset is heavily biased toward common driving environments This means the model will likely perform best in bright, clear, urban or highway settings but may struggle in low-light, adverse weather, or rare scenes like tunnels and parking lots. Rare classes such as train, bike, rider, and motor might face missed detections. 

---

## Insights on Data Collection and Implications

The analysis confirms that this dataset was recorded from **vehicle-mounted cameras**, most from regular cars driving in realistic traffic conditions.  
The forward-facing, object angle, the domiance of city and highway scenes, trending more towards the day all reflect natural human driving behavior.  
These characteristics make the dataset relevant for training autonomous driving perception systems, as it captures the **true common nature** of such systems but however, the dataset is biased not giving equal importance to all scenarios for eg:

- Very low foggy-weather samples for all classes and scenarios like these
- Missing or extremely rare object–condition combinations (eg: bike in snowy weather, train at night).
- Non-uniform placement for certain classes, makes a positional bias.

The individual class analysis tells us several patterns:
- **Traffic lights** and **traffic signs** are smaller, positioned higher in the frame.
- **Cars** dominate all conditions and have strong co-occurrence with many other classes.
- **Train** is nearly absent, making generalization to rail-crossing scenarios unlikely.
- **Bike** and **rider** occur in urban scenes, often occluded by cars or other vehicles.
- **Bus** and **truck** appear more in highway and city street contexts.

For more quality analysis, we could use **fiftyone tool** to find annotation errors and qualitative anomalies.

---
