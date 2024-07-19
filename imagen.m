
fid=fopen('h_Asal.bin','r');
A=fread(fid,[1232 1028],'uint8');
%A=fread(fid,[1028 1232],'uint8');

spy(A')
fclose(fid);