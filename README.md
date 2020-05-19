# layer_cake
**PNG layering and colorizing**

Layer cake is an application developed in python, it's purpose at time of development is to supply PNG alteration for the 'vivacity.online' avatar builder.

Upon initialization, the `Cake` class requires a list of `PNG` locations `[<layers>, ]`. 

`Cake.bake([<layers>, ])` method will create a single `PNG` image constructed by layering each of the provided layers on top of each other.


The `Decorator` class is used to add gradient color to any greyscale `PNG`. 
