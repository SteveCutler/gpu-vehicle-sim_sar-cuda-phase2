# Serial Implementation for later CUDA Porting

TO DO:

-color code goals and vehicles together
-shrink vehicle size
-implement some measurment of error mechanism to compare GPU version against
-create a CPU benchmark

For multiple vehicles need to fix spawn positions to not spawn near target

Naive vehicle advancement is finally working, I got stuck making CUDA classes that didn't work

Naive vehicle advancement and pscan now working

//maybe I should re-write the particle scan to be a vector of 0s, they flip to a 1 if hit and so we never count the same amount twice -> reduce through warp level masks

//particle update flow working
