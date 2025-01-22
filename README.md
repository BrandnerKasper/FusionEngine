# Project Generative Neural Rendering

We want to abstract the traditional renderer of a game engine to a neural network!
We call it project generative neural rendering.

For that we need to check multiple goals.

## Video Super Resolution

- [x] Get RocM version of Pytorch cpp running
- [x] Read in image data
- [x] Get Custom dataloader running
- [x] check if LR and HR are the right pair
- [x] Test network on dummy data -> to check inference speed
- [x] Use a log in combi with the print or just the log, so I can save my training runs in a file
- [x] (random) crop my images to a certain size (so I can stack them together and get a bigger batch size going!)
- [x] abstract my code into useful file structure
- [x] Try to get a simple Single Image super resolution network running
- [ ] Try to get video super res running

## Game Engine
- [ ] Write a simple renderer with either OpenGL or Vulcan
- [ ] Abstract the renderer to use game objects with a Transform and some text based information for rendering said object
- [ ] Make very simple example of rotating cube (maybe multiple) and see if we can use a neural network to substitute the rendering

## Dataset
- [ ] Maybe we should generate the dataset first -> it would be images based of rendered game object(s) and the text data in some form