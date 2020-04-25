img = imread('images/dog_200.png');
img = img/255;

sf = 4.5;
wl = 200/sf;


g = gabor(wl,0);


filt_img = imgaborfilt(img, wl, 0);

figure()
subplot(1,3,1)
imagesc(real(g.SpatialKernel))
subplot(1,3,2)
imagesc(filt_img)
subplot(1,3,3)
imagesc(img)