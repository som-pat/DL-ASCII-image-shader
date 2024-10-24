# GAN ASCII Image render
 Ascii images created and corrected with GAN



#TODO Block-based Processing for Cohesion:

As discussed earlier, splitting the image into larger blocks (e.g., 64x64)
and assigning a representative edge character to each block can improve continuity.
You can calculate a histogram of gradient angles within each block, 
then use the most frequent direction to assign a character to that block.

#TODO replace np.max with histogram
