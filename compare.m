img = imread('images/dog_200.png');
%img = img/255;

sf = 4.5;
wl = 200/10.2;
ori = 360-59;


g = gabor(wl,ori);


[mag, phase] = imgaborfilt(img, wl, ori);
%filt_img = mag .* sin(phase)

figure()
subplot(1,3,1)
imagesc(real(g.SpatialKernel))
subplot(1,3,2)
imagesc(mag)
subplot(1,3,3)
imagesc(img)


% plot the kernel size chose by lambda