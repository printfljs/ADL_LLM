**Project Description**

This project includes two methods for segmenting time-series data: one based on time windows and the other based on activity windows, with implementations for different datasets.

**Environment Requirements**

-   Python 3.19.2

**Environment Variable Configuration**

Create a `.env` file in the root directory and add your API key:

```
OPENAI_API_KEY=your_api_key_here
```

**Code Description**

**Time Window-Based Segmentation**

-   Path: `time_window_based/*.ipynb`
-   Description: This Jupyter Notebook contains code for segmenting time-series data using fixed time windows.
-   Files ending in '_A' are from Dataset A, the primary dataset

**Activity Window-Based Segmentation**

-   Path: `activity_window_based/*.ipynb`
-   Description: This Jupyter Notebook contains code for segmenting data based on activity windows
-   Files ending in '_A' are from Dataset A, the primary dataset
