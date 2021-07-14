IMAGE-Classification

Tech/Fameworks used: OpenCV, Keras/TensorFlow, OpenCV and Deep Learning

Model that recognizes whether an image provided is a selfie(indoor and outdoor),pose(indoor and outdoor) or an image without people


----------------------------------------------------------------------------------------------------

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Personal Photo Classifier\n",
    "#### It classifies Selfies, Poses, Photos"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Table of content\n",
    "1. [Introduction](#introduction)\n",
    "2. [Dataset creation](#Dataset_creation)\n",
    "    1. [Data collection](#cata_collection)\n",
    "    2. [Preview of dataset](#Preview_of_dataset)\n",
    "3. [Training the model](#Training_the_model)\n",
    "4. [Results](#Results)\n",
    "5. [The team](#The_team)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Introduction <a name=\"introduction\"></a>\n",
    "Performed in Brainster - Data Science Bootcamp.\n",
    "\n",
    "We got an Assignment to Classify five type of images:\n",
    "* Indoor Selfie\n",
    "* Outdoor Selfie\n",
    "* Indoor Pose\n",
    "* Outdoor Pose\n",
    "* Photos Without Human"
   ]
  },
  {
   "attachments": {
    "1.jpg": {
     "image/jpeg": "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"
    },
    "2.jpg": {
     "image/jpeg": "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"
    },
    "3.jpg": {
     "image/jpeg": "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"
    },
    "4.jpg": {
     "image/jpeg": "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"
    },
    "5.jpg": {
     "image/jpeg": "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"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Dataset creation <a name=\"Dataset_creation\"></a>\n",
    "Dataset creation is really important to achieve a good accuracy on Image Classification tasks.\n",
    "### 2.A. Data Collection <a name=\"Data_collection\"></a>\n",
    "We collected data from internet, we tried to use caution to not violate any copyrights.\n",
    "\n",
    "The collection part was not easy, tried plenty of different methods.\n",
    "### 2.B. Preview of dataset <a name=\"Preview_of_dataset\"></a>\n",
    "| **Class**|**Example 1**|**Example 2** |**Example 3**|**Example 4**|**Example 5**|  \n",
    "|:---:|:---:|:---:|:---:|:---:|:---:|\n",
    "|**Indoor Selfie**|![1.jpg](attachment:1.jpg)|![2.jpg](attachment:2.jpg)|![3.jpg](attachment:3.jpg)|![4.jpg](attachment:4.jpg)|![5.jpg](attachment:5.jpg)|\n",
    "|**Outdoor Selfie**|![1.jpg](attachment:1.jpg)|![2.jpg](attachment:2.jpg)|![3.jpg](attachment:3.jpg)|![4.jpg](attachment:4.jpg)|![5.jpg](attachment:5.jpg)||\n",
    "|**Indoor Pose**    |![1.jpg](attachment:1.jpg)|![2.jpg](attachment:2.jpg)|![3.jpg](attachment:3.jpg)|![4.jpg](attachment:4.jpg)|![5.jpg](attachment:5.jpg)|\n",
    "|**Outdoor Pose**  |![1.jpg](attachment:1.jpg)|![2.jpg](attachment:2.jpg)|![3.jpg](attachment:3.jpg)|![4.jpg](attachment:4.jpg)|![5.jpg](attachment:5.jpg)|\n",
    "|**Without Human** |![1.jpg](attachment:1.jpg)|![2.jpg](attachment:2.jpg)|![3.jpg](attachment:3.jpg)|![4.jpg](attachment:4.jpg)|![5.jpg](attachment:5.jpg)|"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Training the model <a name=\"Training_the_model\"></a>\n",
    "xyz\n",
    "### 3.A. xyz <a name=\"xyz\"></a>\n",
    "xyz\n",
    "### 3.B. xyz <a name=\"xyz\"></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Results <a name=\"Results\"></a>\n",
    "xyz\n",
    "### 4.A. xyz <a name=\"xyz\"></a>\n",
    "xyz\n",
    "### 4.B. xyz <a name=\"xyz\"></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. The team <a name=\"The_team\"></a>\n",
    "* [Sabina Dzafic]()\n",
    "* [Sushma Harsh ???](https://github.com/sushmavenu)\n",
    "* [Daniel Varga](https://github.com/IndaPerpetuum)\n",
    "\n",
    "\n",
    "\n",
    "This project was supervised by [Igor Trpevski]()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#RULES\n",
    "\n",
    "#Capital does not matter, Link naming = Link_naming\n",
    "#Alphabetical ordering in team"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

