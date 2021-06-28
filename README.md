# Sign Language Detection Using Deep Learning

<b> Note : </b> The complete code along with pretrained model is availbale at [this](https://drive.google.com/file/d/1asp49Y5LbjCnRxetISSnXsfn-cHYaRLP/view?usp=sharing) link 

## Why?

1. Communication
  * Communication comes from the Latin word communicare, meaning "to share“
  * It helps us humans share our feeling and create bonds
  * But what about mute people?
  * They must not be deprived the ability to bond just because they can’t talk.
2. Sign Language 
  * Mute people predominantly use sign language for their daily communication
  * But not everyone can understand sign language
  * Viewing a gesture from another perspective makes it difficult to be understood
  * Since every finger position and movement will not be observable.

## How?
#### Our Idea 
- Cameras can be attached to the head or chest of a mute person
- The camera observe the hand gesture and recognises it using Deep learning
- Using text-to-speech software the recognised gesture can be converted to audio
- The cameras attached will in no way hinder their daily activities

#### Our Model
- We have written our model using Pytorch library and trained it using GPU
- The model was able to recognise the signs with 96-97% accuracy.
- We have currently deployed our model to a desktop app
- The app takes an image as an input and gives out the predicted character 
- The model can identify alphabets, numbers and an underscore character used as space

#### Data Set Used: [Sign Language Gesture Images Dataset - Unique 37 Hand Sign Gestures](https://www.kaggle.com/ahmedkhanak1995/sign-language-gesture-images-dataset)

#### Model Summary 
> 
>        Layer (type)              Output Shape         Param 
> 
>            Linear-1               [-1, 1024]       7,681,024
>             ReLU-2                [-1, 1024]               0
>            Linear-3               [-1, 2048]       2,099,200
>             ReLU-4                [-1, 2048]               0
>            Linear-5               [-1, 2048]       4,196,352
>              ReLU-6               [-1, 2048]               0
>            Linear-7               [-1, 2048]       4,196,352
>              ReLU-8               [-1, 2048]               0
>            Linear-9               [-1, 1024]       2,098,176
>             ReLU-10               [-1, 1024]               0
>           Linear-11                [-1, 512]         524,800
>             ReLU-12                [-1, 512]               0
>           Linear-13                 [-1, 37]          18,981
> ------------------------------------------------------------------
> Total params: 20,814,885;
> Trainable params: 20,814,885; 
> Non-trainable params: 0
> 
> Input size (MB): 0.03;
> Forward/backward pass size (MB): 0.13;
> Params size (MB): 79.40;
> 
> Estimated Total Size (MB): 79.56
