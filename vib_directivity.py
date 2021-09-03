#Python program to look at the directivity from a set of vibrators (array)
#Thomas Elboth - Shearwater geoservice Feb-Mar 2021

#pip install numpy streamlit matplotlib plotly

import streamlit as st
import numpy as np
import time
import math, io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

import pickle
from scipy import signal
from spectrum import *  #to do multi-taper


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def binarySearch(data, val):
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if data[mid] < val:
            lo = mid + 1
        elif data[mid] > val:
            hi = mid - 1
        else:
            best_ind = mid
            break
        # check if data[mid] is closer to val than data[best_ind]
        if abs(data[mid] - val) < abs(data[best_ind] - val):
            best_ind = mid
    return best_ind



def plot_matplotlib_birdseye(f, f_out, txt, R=9000):

        [no_dir, no_dir, no_freq]  = f_out.shape
        max_val = np.amax(f_out)
        min_val = np.amin(f_out)
        if ((max_val <= np.pi) and (min_val >= -np.pi)): #we have a phase plot
            vmin=-np.pi
            vmax= np.pi
            col_bar_ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi ]
        else:                                           #we have an amplitude plot
            vmin=0
            vmax=round(max_val)
            f_out=f_out.clip(min=0)
            col_bar_ticks = np.floor(np.linspace(vmin,vmax,8))

        F=np.squeeze(f_out[:,:,f])

        theta = np.linspace(180, 360, no_dir)
        r = np.linspace(180,360,no_dir)

        r, theta = np.meshgrid(r, r)
        fig, ax = plt.subplots()
        ax.contourf(theta.T,r.T,F.T,  vmin=vmin, vmax=vmax, cmap='jet', levels=100)
        ax.set_xlabel(txt)

        return fig


def plot_matplotlib_polar(f_out, max_freq, text, f_out_full):
    max_val = np.amax(f_out_full)
    min_val = np.amin(f_out_full)

    if ((max_val <= np.pi) and (min_val >= -np.pi)): #we have a phase plot
        vmin=-np.pi
        vmax= np.pi
        col_bar_ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi ]
    else:                                           #we have an amplitude plot
        vmin=0
        vmax=round(max_val)
        f_out=f_out.clip(min=0)
        col_bar_ticks = np.floor(np.linspace(vmin,vmax,8))
    [no_dir, samps]  = f_out.shape

    x = np.linspace(-np.pi/2, np.pi/2, no_dir)
    y = np.linspace(0,max_freq,samps)
    [THETA,RR] = np.meshgrid(x,y)

    # #try matplotlib - which is more similar to the matlab code
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    cntr1=ax.contourf(THETA+3*np.pi/2, RR, f_out.T, vmin=vmin, vmax=vmax, cmap='jet', levels=100 )
    my_ticks=[180*np.pi/180, 225*np.pi/180, 270*np.pi/180, 325*np.pi/180, 360*np.pi/180]
    ax.set_thetamin(180)
    ax.set_thetamax(360)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xticks(my_ticks)
    ax.set_xlabel(text)
    fig.colorbar(cntr1, ax=ax, orientation='vertical',  shrink=.6, pad=0.1, ticks=col_bar_ticks )

    return fig

def average(lst):
    return sum(lst) / len(lst)

#Timedelay a signal by n samples
def timedelay_a_signal(waveform, tDelayInSamples):
    # 1. Take the FFT
    fftData = np.fft.fft(waveform)
    # 2. Construct the phase shift
    N = fftData.shape[0]
    k = np.linspace(0, N-1, N)
    timeDelayPhaseShift = np.exp(((-2*np.pi*1j*k*tDelayInSamples)/(N)) + (tDelayInSamples*np.pi*1j))
    # 3. Do the fftshift on the phase shift coefficients
    timeDelayPhaseShift = np.fft.fftshift(timeDelayPhaseShift)
    # 4. Multiply the fft data with the coefficients to apply the time shift
    fftWithDelay = np.multiply(fftData, timeDelayPhaseShift)
    # 5. Do the IFFT
    shiftedWaveform = np.fft.ifft(fftWithDelay)
    return shiftedWaveform

#Here we get a file object that (hopefully) contains a source signature.
#We need to extract the signature - based on filename etc, etc (try  - try and try)
def get_signature_from_file(uploaded_file):
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        if(".far" in uploaded_file.name): #Nucleus far-field type of file
            try:
                print("We have a Nucleus file - to read")
                raw_text = str(uploaded_file.read(),"utf-8") # read the file content
                #find the sampling interval:
                indx = raw_text.find('Sample interval (ms)        :')
                si = raw_text[indx+len('Sample interval (ms)        :'): indx+len('Sample interval (ms)        :')+10 ].strip()
                si = si.split()

                #skip all headers (knowing that the last header word is "horizontally")
                indx = raw_text.find('horizontally')
                text = raw_text[indx+len('horizontally'):len(raw_text)]

                text = text.split() #split the text on space
                signature=[float(i) for i in text]
                #print("1: The sampling interval (SI) is:", si)
                return [signature, float(si[0])]
            except:
                st.write('Unable to reat the nucleus file? Something went wrong?', uploaded_file.name,'.')
        elif(".data" in uploaded_file.name):
            print('Reading vib-file', uploaded_file.name)
            fname = "Vib_Database/"+uploaded_file.name
            return readSignature2(fname)


#read the fle directly - mainly for testing
def readSignature(filename:str):
    f = open(filename, 'r')
    raw_text = f.read()
    f.close()
    try:
        indx = raw_text.find('Sample interval (ms)        :')
        si = raw_text[indx+len('Sample interval (ms)        :'): indx+len('Sample interval (ms)        :')+10 ].strip()
        si = si.split()

        #skip all headers (knowing that the last header word is "horizontally")
        indx = raw_text.find('horizontally')
        text = raw_text[indx+len('horizontally'):len(raw_text)]

        text = text.split() #split the text on space
        signature=[float(i) for i in text]
        #print("0: The sampling interval (SI) is:", si)
        return [signature, float(si[0])]
    except:
        st.write('This was NOT a nucleus signature file...', filename,'.')
        return readSignature2(filename)

def readSignature2(filename:str):
    try:
        df = pd.read_csv(filename,   header=None, delimiter=r"\s+", engine='python')

        time         = df.iloc[:,0]
        double_integral   = df.iloc[:,1]
        NS           = df.iloc[:,2]
        amplitude_NS = df.iloc[:,5]
        u_freq       = df.iloc[:,3]
        applied_phaseroll = df.iloc[:,4]

        si = abs(time[1]-time[0])
        #print("si=",si)
        return [double_integral, si]

    except:
        st.write('Could not read this file....', filename)


def plot3d_directivity_and_phase(signature, sampling_interval, inline_pos, crossline_pos, depth_pos, no_of_vibs, max_freq,  ff_pos, include_ghost, frequency_birdseye):
    t0 = time.time()
    #compute the center of source (both inline and crosline) and adjust source pos accordingly
    inline_pos    = inline_pos - average(inline_pos)
    crossline_pos = crossline_pos - average(crossline_pos)

    #add the ghost to the primary signal
    no_dir=40  #need at least 40 to get the high resilution for the directive source
    R=ff_pos
    SI = sampling_interval[0]
    speed_of_sound=1500
    ghost_reflection_coeff=0.0
    if include_ghost==True:
        ghost_reflection_coeff=-0.99

    #looking in the inline (x) and crossline(y) direction first
    theta= np.linspace(-np.pi/2, np.pi/2, no_dir)
    phi=   np.linspace(-np.pi/2, np.pi/2, no_dir)

    #precompute some stuff to save time
    cos_theta = np.zeros((no_dir))
    sin_theta = np.zeros((no_dir))
    for k in range(0, no_dir):
        cos_theta[k] = math.cos(theta[k])
        sin_theta[k] = math.sin(theta[k])

    sig=   np.zeros((no_dir, no_dir, len(signature[0,:])))
    f_out= np.zeros((no_dir, no_dir, len(np.fft.rfft(signature[0,:])) ))
    #phase =np.zeros((no_dir, no_dir, len(np.fft.rfft(signature[0,:])) ))

    factor = (1000/(speed_of_sound*SI))
    if (factor>10): ##we have vib-data sampled differently
        factor = factor/1000

    my_window = np.hamming(2*len(f_out[0,0,:])-1)

    for i in range(0, no_dir):                  #PHI
        y= R*math.sin(phi[i])
        tmp0 = R*math.cos(phi[i])
        for k in range(0, no_dir):              #THETA
            z= cos_theta[k]*tmp0
            x= sin_theta[k]*tmp0
            z2=z*z

            for j in range(0, no_of_vibs):

                xm = x+inline_pos[j]
                xm2= xm*xm
                ym = y+crossline_pos[j]
                ym2= ym*ym

                #the geometrical delay + the delay from sources at different depths being fired with a delay
                Delay_in_samps_geom = (R-math.sqrt(xm2+ym2+z2))*factor                  + depth_pos[j]*factor
                #the ghost delay (mirrored on the water-layer --> 2z) +  the dalay due to sources at different depths
                Delay_in_samps_ghost = (R-math.sqrt(xm2+ym2+(z+2*depth_pos[j])**2))*factor + depth_pos[j]*factor

                #apply the delays to the primary and ghost - and add them together
                primary =                      np.real(timedelay_a_signal(signature[j,:], Delay_in_samps_geom) ) #This works for non-integer delays
                ghost = ghost_reflection_coeff*np.real(timedelay_a_signal(signature[j,:], Delay_in_samps_ghost) )

                sig[k,i,:] = sig[k,i,:] + primary+ghost

            tmp = np.fft.rfft(sig[k,i,:]*my_window)
            f_out[k,i,:] = ( abs(tmp) )
            #phase[k,i,:] = np.angle(tmp)

    #cut the frequency range to [0, max_freq]
    samps = round(len(signature[0])*max_freq/(1000/SI))
    if(samps < 10):
        samps =round(len(sig[0,0,:])*max_freq/4000)

    f_out = 20*np.log10(f_out[:,:,0:samps])
    #phase = phase[:,:,0:samps]

    #make the various plots:
    #) Inline view
    fig3 = plot_matplotlib_polar(f_out[:,round(no_dir/2),:], max_freq, 'Inline amplitude', f_out  )
    #fig3 = plot_plotly_polar(f_out[:,round(no_dir/2),:], max_freq, 'Inline amplitude', f_out  )
    #fig4 = plot_matplotlib_polar(phase[:,round(no_dir/2),:], max_freq, 'Inline phase', phase)

    #1) Crossline view:
    fig1 = plot_matplotlib_polar(f_out[round(no_dir/2),:,:], max_freq, 'Crossline amplitude', f_out)
    #fig2 = plot_matplotlib_polar(phase[round(no_dir/2),:,:], max_freq, 'Crossline phase', phase)

    #print("The shape",f_out.shape, f_out[round(no_dir/2),:,:].shape, f_out[:,round(no_dir/2),:].shape)

    txt = "Birdseye - Amplitude at "+str(frequency_birdseye)+" Hz"
    f= round(len(signature[0])*frequency_birdseye/(1000/SI))
    print('f=',f)
    if(f<3): #a vib with another SI

        f= round((2*samps/250)*frequency_birdseye)

    fig5 = plot_matplotlib_birdseye(f, f_out, txt, R=9000)
    t1 = time.time()
    print("This took:",str(round(t1-t0)),"sec.")
    return [fig1, fig3, fig5]


def plot_array_directivity():
    st.title('Source/Vibrator array analysis ')

    my_expander1 = st.expander("General parameters:", expanded=True)
    with my_expander1:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            no_of_vibs = st.number_input('Number of vibrators in one array:',1,9,1,1)
        with col4:
            max_freq =  st.number_input('Select maximum frequency',1,250,150,1)
            #inline_view= st.selectbox('Produce inline view plot',(True,False))
        with col2:
            include_ghost= st.selectbox('Include the source ghost',(True,False))
            #crossline_view= st.selectbox('Produce crossline view plot',(False,True))
        with col3:
            ff_pos = st.number_input('Far field distance (m)',100,9000,9000,100)
        with col5:
            frequency_birdseye = st.number_input('Select birdseye frequency',1,250,40,1)

        inline_pos = np.zeros((50))
        crossline_pos = np.zeros((50))
        depth_pos = np.zeros((50))
        source_type = ["" for x in range(50)]

        signature=[]
        sampling_interval=[]

    my_expander2 = st.expander("The individual source elements:", expanded=True)
    with my_expander2:
        col1, col2, col3, col4, col5 = st.columns(5)
        key_count=0

        try:
            inline_pos_default = np.load("inline.npy")
        except:
            inline_pos_default=np.zeros((50)) #up to 50 source elements supported
            np.save('inline.npy', inline_pos_default)

        try:
            crossline_pos_default = np.load("xline.npy")
        except:
            crossline_pos_default=np.zeros((50)) #up to 50 source elements supported
            np.save('xline.npy', crossline_pos_default)

        try:
            depth_pos_default = np.load("depth.npy")
        except:
            depth_pos_default=5*np.ones((50)) #up to 50 source elements supported
            np.save('depth.npy', depth_pos_default)

        try:
            with open ('type.pkl', 'rb') as fp:
                type_default = pickle.load(fp)
        except:
            type_default=[]
            for k in range(0,50):
                type_default.append('predef')
            with open('type.pkl', 'wb') as fp:
                pickle.dump(type_default, fp)

        with col1:
            st.write('Inline position (m)')
            for i in range(0, no_of_vibs):
                key_count = key_count+1 #to get a uniq key for every entry
                inline_pos[i] = st.number_input('Vib #'+str(i+1)+':',-3000.0,3000.0,inline_pos_default[i], 0.5, None, key=key_count)
                inline_pos_default[i] = inline_pos[i]
            #save the pos in each iteration
            np.save('inline.npy', inline_pos)

        with col2:
            st.write('Crossline position (m)')
            for i in range(0, no_of_vibs):
                key_count = key_count+1 #to get a uniq key for every entry
                crossline_pos[i] = st.number_input('Vib #'+str(i+1)+':',-3000.0,3000.0,crossline_pos_default[i], 0.5, None, key=key_count)
                crossline_pos_default[i] =crossline_pos[i]
            #save the pos in each iteration
            np.save('xline.npy', crossline_pos)

        with col3:
            st.write('Depth (m)')
            for i in range(0, no_of_vibs):
                key_count = key_count+1 #to get a uniq key for every entry
                depth_pos[i] = st.number_input('Vib #'+str(i+1)+':',0.0,50.0,depth_pos_default[i],0.5,None ,key=key_count)
                depth_pos_default[i] = depth_pos[i]
            #save the pos in each iteration
            np.save('depth.npy', depth_pos)

        with col4:
            st.write('Source_type')
            for i in range(0, no_of_vibs):
                key_count = key_count+1 #to get a uniq key for every entry
                source_type[i]=  st.selectbox('Type of input:',[type_default[i], 'predef','LF','HF_omni','HF_dir','From file'],key=key_count)
                type_default[i] = source_type[i]
            with open('type.pkl', 'wb') as fp:
                pickle.dump(type_default, fp)


        with col5:
            #TODO - enable users to read the full Nucleus DB....
            st.write('Signature file upload')
            signature=[] #these are just some default signatures to read in.... tor testing
            dir_name="Gun_Database"
            file_gun=[ os.path.join(dir_name, "45_cluster_6m.far"),
            os.path.join(dir_name,"45_cluster_6m.far"),
            os.path.join(dir_name,"45_cluster_6m.far"),
            os.path.join(dir_name,"250_cluster_9m.far"),
            os.path.join(dir_name,"180_single_6m.far"),
            os.path.join(dir_name,"100_cluster_9m.far"),
            os.path.join(dir_name,"100_cluster_6m.far"),
            os.path.join(dir_name,"80_single_6m.far")]

            dir_name="Vib_Database"
            file_hf_dir=[os.path.join(dir_name,"v17_OUTPUT_DOWNSWEEP_0deg_phase_sec_disp_acc_Hz_rollrad_VIBGROUP_1_OBN_DIR.data"),
                       os.path.join(dir_name,"v17_OUTPUT_DOWNSWEEP_180deg_phase_sec_disp_acc_Hz_rollrad_VIBGROUP_1_OBN_DIR.data"),
                       os.path.join(dir_name,"v17_OUTPUT_DOWNSWEEP_90deg_phase_sec_disp_acc_Hz_rollrad_VIBGROUP_1_OBN_DIR.data"),
                       os.path.join(dir_name,"v17_OUTPUT_DOWNSWEEP_270deg_phase_sec_disp_acc_Hz_rollrad_VIBGROUP_1_OBN_DIR.data")]

            file_hf_omni=[os.path.join(dir_name,"v17_OUTPUT_DOWNSWEEP_0deg_phase_sec_disp_acc_Hz_rollrad_VIBGROUP_1_OBN_OMNI.data"),
                       os.path.join(dir_name,"v17_OUTPUT_DOWNSWEEP_180deg_phase_sec_disp_acc_Hz_rollrad_VIBGROUP_1_OBN_OMNI.data"),
                       os.path.join(dir_name,"v17_OUTPUT_DOWNSWEEP_90deg_phase_sec_disp_acc_Hz_rollrad_VIBGROUP_1_OBN_OMNI.data"),
                       os.path.join(dir_name,"v17_OUTPUT_DOWNSWEEP_270deg_phase_sec_disp_acc_Hz_rollrad_VIBGROUP_1_OBN_OMNI.data")]

            file_lf_omni=[os.path.join(dir_name,"v17_OUTPUT_DOWNSWEEP_0deg_phase_sec_disp_acc_Hz_rollrad_VIBGROUP_2_OBN_OMNI.data"),
                       os.path.join(dir_name,"v17_OUTPUT_DOWNSWEEP_180deg_phase_sec_disp_acc_Hz_rollrad_VIBGROUP_2_OBN_OMNI.data"),
                       os.path.join(dir_name,"v17_OUTPUT_DOWNSWEEP_90deg_phase_sec_disp_acc_Hz_rollrad_VIBGROUP_2_OBN_OMNI.data"),
                       os.path.join(dir_name,"v17_OUTPUT_DOWNSWEEP_270deg_phase_sec_disp_acc_Hz_rollrad_VIBGROUP_2_OBN_OMNI.data")]

            sampling_interval=[]
            max_len = 0
            for i in range(0, no_of_vibs):
                key_count = key_count+1 #o get a uniq key for every entry
                if(source_type[i]=="From file"):
                    file = st.file_uploader("Choose a file (on server)", key=key_count)
                    try:
                        [sig, si] = get_signature_from_file(file)
                    except:
                        st.write('You need to pick a source signature file')
                        return 0

                if(source_type[i]=='HF_omni'):
                    [sig, si] = readSignature2(file_hf_omni[i])

                if(source_type[i]=='HF_dir'):
                    [sig, si] = readSignature2(file_hf_dir[i])

                if(source_type[i]=='LF' ):
                    [sig, si] = readSignature2(file_lf_omni[i])

                if(source_type[i]=='predef'): #just default some air-gun files
                    [sig, si] = readSignature(file_gun[i])

                signature.append(sig)
                sampling_interval.append(si)

                if len(sig)>max_len:
                    max_len = len(sig)

            #go through - and check that all signatures have the same length - if not - we pad with zeros
            for i in range(0, no_of_vibs):
                #print("men and maxlen:",len(signature[i]),max_len)
                try:
                    if (len(signature[i]) < max_len):
                        tmp = np.zeros((max_len))
                        tmp[0:len(signature[i])] = signature[i]
                        signature[i] = tmp
                except:
                    print("hmmm")

            #convert the signature list to a numpy array (for speed)
            signature = np.asarray(signature, dtype=np.float32)
            #print("Signature.shape=", signature.shape)

    my_expander3 = st.expander("Plot source layout:", expanded=False)
    with my_expander3:
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=inline_pos[0:no_of_vibs], y=crossline_pos[0:no_of_vibs], mode='markers',  marker_size=12, marker_symbol='star'))
            fig.update_layout(title='Source layout (birdseye)',  xaxis_title='Inline (m)',yaxis_title='Xrossline (m)')
            st.plotly_chart(fig)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=inline_pos[0:no_of_vibs], y=depth_pos[0:no_of_vibs], mode='markers',  marker_size=12, marker_symbol='star'))
            fig.update_layout(title='Source layout (inline-depth)',  xaxis_title='Inline (m)',yaxis_title='Depth (m)')
            st.plotly_chart(fig)
        with col3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=crossline_pos[0:no_of_vibs], y=depth_pos[0:no_of_vibs], mode='markers',  marker_size=12, marker_symbol='star'))
            fig.update_layout(title='Source layout (crossline-depth)',  xaxis_title='Crossline (m)',yaxis_title='Depth (m)')
            st.plotly_chart(fig)

    my_expander4 = st.expander("Plot individual element signatures:", expanded=False)
    with my_expander4:
        col1, col2 = st.columns(2)
        with col1:
            #plot the t-x signatures of the individual sources - just to to get started
            fig = go.Figure()
            if(len(signature)>0 and len(sampling_interval)>0):
                t=np.linspace(0,len(signature[0])*sampling_interval[0], len(signature[0]))
                if(t[-1] < 20):
                    t=t*1000

                for i in range(0, len(signature)):
                    fig.add_trace(go.Scatter(x=t, y=signature[i], mode='lines' ))

                txt = ('Source time-amplitude plot')
                fig.update_layout(title=txt,  xaxis_title='Milliseconds',yaxis_title='Amplitude')
                st.plotly_chart(fig)

                #make a spectrogram
                for i in range(0, len(signature)):
                    if(sampling_interval[i]<0.02): #we assume this is a vib - and not a gun
                        [f, t, Sxx] = signal.spectrogram(x=signature[i], fs=1/sampling_interval[i], nfft=256*16)
                        maxFreq=250
                        my_len = round(maxFreq*Sxx.shape[0]/(1.0/(2*sampling_interval[0])))

                        f=f[0:my_len]
                        Sxx=Sxx[0:my_len,:]
                        Sxx=Sxx/ max(map(max, Sxx)) #normalize

                        fig = go.Figure(data=go.Heatmap(x=t, y=f, z=Sxx))
                        txt = 'Spectrogram: Normalized amplitude of source #'+str(i+1)+'.'
                        fig.update_layout(xaxis_title = 'Time (sec)', yaxis_title='Frequency (Hz)', title=txt)
                        st.plotly_chart(fig)

                        #and the autocorrelation of the sweep:
                        corr= np.correlate(signature[i], signature[i], mode="full")
                        corr=corr/max(corr)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=corr, mode='lines',name='auto-correlation' ))
                        txt = 'Autocorrelation of vibrator sweep #'+str(i+1)+'.'
                        fig.update_layout(xaxis_title = 'Sample', yaxis_title='Correlation coeff [-1,1]', title=txt)
                        st.plotly_chart(fig)


        with col2:
            fig = go.Figure()
            if(len(signature)>0 and len(sampling_interval)>0):
                t=np.linspace(0,1000/sampling_interval[0], len(signature[0]))
                samps = round(len(signature[0])*250/(1000/sampling_interval[0]))
                if(t[-1] < 20):
                    t=t*1000

                for i in range(0, len(signature)):

                    #multitaper spectral estimate
                    A, weights, eigenvalues = pmtm(signature[i], NW=2.5,k=4)
                    A = 20*np.log10(np.mean(abs(A) * np.transpose(weights), axis=0))
                    #cut out the 0-250Hz range
                    A=A[0: round(len(A)*250/4000)]

                    #normal fft for comparing
                    #B= 20*np.log10(abs(np.fft.rfft(signature[i]*np.hamming(len(signature[i])))))

                    if(samps<100):
                        t= np.linspace(0, 0.5/sampling_interval[i], len(signature[i]))
                        samps = round(len(signature[i])*250/2000)

                    ##plot the fft results
                    #fig.add_trace(go.Scatter(x=t[0:samps], y=B[0:samps], mode='lines', name='fft' ))

                    #plot the multitaper results - which is pow2 long....
                    fig.add_trace(go.Scatter(x=np.linspace(0,250, len(A)),y=A, mode='lines',name='MultiTaper' ))

                txt = ('Multitaper spectral amplitude estimation')
                fig.update_layout(title=txt,  xaxis_title='Frequency (Hz)',yaxis_title='Amplitude (dB)')
                st.plotly_chart(fig)

                #make a spectrogram as well
                for i in range(0, len(signature)):
                    if(sampling_interval[i]<0.02): #we assume this is a vib - and not a gun
                        [f, t, Sxx] = signal.spectrogram(x=signature[i], fs=1/sampling_interval[i], nfft=256*16)
                        maxFreq=250
                        my_len = round(maxFreq*Sxx.shape[0]/(1.0/(2*sampling_interval[0])))
                        f=f[0:my_len]
                        Sxx=Sxx[0:my_len,:]
                        Sxx=Sxx/ max(map(max, Sxx)) #normalize

                        fig = go.Figure(data=go.Heatmap(x=t, y=f, z=np.log10(Sxx)))
                        txt = 'Spectrogram: Normalized amplitude of source #'+str(i+1)+' in dB.'
                        fig.update_layout(xaxis_title = 'Time (sec)', yaxis_title='Frequency (Hz)', title=txt)
                        st.plotly_chart(fig)

    my_expander5 = st.expander("Directivity plots:", expanded=True)

    delay=[]
    if len(sampling_interval)==no_of_vibs:
        col1, col2, col3 = st.columns(3)
        with col1:
            plot_2d = st.button("Plot source directivity")
            if (plot_2d):
                [fig1, fig3, fig5]=plot3d_directivity_and_phase(signature, sampling_interval, inline_pos, crossline_pos, depth_pos, no_of_vibs, max_freq, ff_pos, include_ghost, frequency_birdseye)
                st.pyplot(fig1)
                #st.pyplot(fig2)
        with col2:
            plot_2d_crossline = st.button("To be added in the future")
            if (plot_2d):
                st.pyplot(fig3)
                #st.pyplot(fig4)
        with col3:
            if (plot_2d):
                st.pyplot(fig5)

    else:
        st.write('Need signatures from all vibes to compute directivity plots')


def main():
    plot_array_directivity()


if __name__ == "__main__":
    main()
