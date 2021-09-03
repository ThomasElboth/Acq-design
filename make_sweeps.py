#python program to make sparse frequency sweeps
#Thomas Elboth - Aug/Sept 2021


import streamlit as st                #this is the gui stuff
import base64                        #for download functionality in streamlit
import numpy as np
import math
import plotly.graph_objects as go
import pandas as pd
from scipy import signal, integrate, interpolate
from scipy.signal import butter, lfilter, hilbert, savgol_filter

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(sep="\t",index=False, header=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


#to filter away any potential HF noise in the data
def butter_bandpass(lowcut, highcut, fs, order=5): #FIR filter = constant phase delay + "stable"
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def plotsweep(signature, sampling_interval, txt, spec=True):
    my_expander = st.expander(txt, expanded=False)
    with my_expander:
        col1, col2 = st.columns(2)
        with col1:
            #plot the t-x signatures of the individual sources - just to to get started
            fig = go.Figure()
            t=np.linspace(0,len(signature)*sampling_interval, len(signature))

            #fig.add_trace(go.Scatter(x=t[900:930], y=signature[900:930], mode='lines' ))
            fig.add_trace(go.Scatter(x=t, y=signature, mode='lines' ))
            fig.update_layout(title=txt,  xaxis_title='Time (sec)',yaxis_title='Amplitude')
            st.plotly_chart(fig)

            if (spec==True):
                #make a spectrogram
                [f, t, Sxx] = signal.spectrogram(x=signature, fs=1/sampling_interval, nfft=256*16)
                maxFreq=250
                my_len = round(maxFreq*Sxx.shape[0]/(1.0/(2*sampling_interval)))

                f=f[0:my_len]
                Sxx=Sxx[0:my_len]
                Sxx=Sxx/ max(map(max, Sxx)) #normalize

                fig = go.Figure(data=go.Heatmap(x=t, y=f, z=Sxx))
                txt = 'Spectrogram: Normalized amplitude'
                fig.update_layout(xaxis_title = 'Time (sec)', yaxis_title='Frequency (Hz)', title=txt)
                st.plotly_chart(fig)

            # #and the autocorrelation of the sweep:
            # corr= np.correlate(signature, signature, mode="full")
            # corr=corr/max(corr)
            # fig = go.Figure()
            # fig.add_trace(go.Scatter(y=corr, mode='lines',name='auto-correlation' ))
            # txt = 'Autocorrelation of vibrator sweep'
            # fig.update_layout(xaxis_title = 'Sample', yaxis_title='Correlation coeff [-1,1]', title=txt)
            # st.plotly_chart(fig)


def envelope(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    amplitude_envelope=savgol_filter(amplitude_envelope, 51, 3)  #just a bit of smoothening
    return amplitude_envelope


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


def get_nominal_sweep_length(f_start, f_end, underlying_freq):
    if(underlying_freq[0]>underlying_freq[-1]):
        underlying_freq=np.flipud(underlying_freq)
    return abs( binarySearch(underlying_freq,f_start) - binarySearch(underlying_freq,f_end))

#huild up a new sweep - one frequency at a time. We use the "underlying freq" as a guide.
def generate_sweep(f_start, f_end, fs, b, underlying_freq, factor=1):

    if(underlying_freq[0]>underlying_freq[-1]):
        underlying_freq=np.flipud(underlying_freq)

    sweep = np.zeros(10*underlying_freq.size) #make it a bit longer than needed
    count=0
    for freq in range(f_start, f_end):

        #determine how long to sweep at each frequency
        len1  = fs/freq    #NQ min #samples at this freq
        len2  = fs/(freq+1)
        #this is how long robert sweept at this freq (SnR sweep...)
        desired_len =round(factor* abs( binarySearch(underlying_freq,freq) - binarySearch(underlying_freq,freq+1)))
        len_avr=np.linspace(len1, len2, desired_len)

        for i in range (desired_len):
            sweep[count] = math.sin( b + 2*math.pi*i/len_avr[i] )
            count=count+1;

        b=b+2*math.pi*(desired_len/len2)  #keeping track of the phase

    return [sweep[0:count], b]  #cut sweep to the correct length

#just applying a cos2 taper. Experiment have indicatrd that a taper length of around 60 sampls probably is enough
#Todo: it might be interesting to use anouther taper that is more differentiable then the cos2 taper, but this si probably not neded
def applyTaper(signal, taper_len=100):
    taper = np.linspace(0,math.pi/2,taper_len)
    for i in range(0, taper_len):
        signal[i] = signal[i]*(1.0-(math.cos(taper[i])*math.cos(taper[i])))                     #beginning
        signal[signal.size-1-i] = signal[signal.size-1-i]*(1.0-(math.cos(taper[i])*math.cos(taper[i]))) #end

    return signal


#when constructing the sweeps - they sometimes have the wrong length - due to rounding errors
def check_sweep_length(sweep, f, nominal_len):
    if(sweep.size==0):
        return [sweep, f]
    while (sweep.size < nominal_len):
        sweep= np.append(sweep, 0)
    while (f.size < nominal_len):
        f=np.append(f, f[-1])
    sweep=sweep[0:round(nominal_len)]  #in case they were too long
    f=f[0:round(nominal_len)]
    return [sweep, f]



def smooth_data_np_cumsum_my_average(arr, span):
    cumsum_vec = np.cumsum(arr)
    moving_average = (cumsum_vec[2 * span:] - cumsum_vec[:-2 * span]) / (2 * span)
    # The "my_average" part again. Slightly different to before, because the
    # moving average from cumsum is shorter than the input and needs to be padded
    front, back = [np.average(arr[:span])], []
    for i in range(1, span):
        front.append(np.average(arr[:i + span]))
        back.insert(0, np.average(arr[-i - span:]))
    back.insert(0, np.average(arr[-2 * span:]))
    return np.concatenate((front, moving_average, back))


#scale the sweep correctly at each frequency - using the amplitude of the NS as a guide
def scaleAmplitudes(sweep, u_freq, amplitude_NS, f):
    amplitude = np.zeros(sweep.size)


    for i in range(0, sweep.size):
         indx=binarySearch(u_freq,f[i])
         amplitude[i] = amplitude_NS[indx]

    if(max(f)<50):  #a lf sweep
        amplitude = smooth_data_np_cumsum_my_average(amplitude, 400)
        amplitude=abs(savgol_filter(amplitude, min(sweep.size,51), 3))
    else:           #a hf sweep
        amplitude=abs(savgol_filter(amplitude, min(sweep.size,2501), 3))
    ##plotsweep(sweep,0.00025,"The the sweep",False)
    ##plotsweep(amplitude,0.00025,"The amplitude",False)
    sweep = np.multiply(sweep,amplitude)
    ##plotsweep(sweep,0.00025,"The the sweep (after multiply)",False)
    #apply the taper here
    #sweep=applyTaper(sweep, 50)
    return sweep

#compute the double derivative - TODO: fix the ends
def my_der2(u, fs):
    out_data = np.zeros(u.size)
    for i in range(2,u.size-1):
        out_data[i] =  u[i+1]-2*u[i]+u[i-1];
    return fs*fs*out_data

#integrate twice, but do a 100x interpolation to increase accuracy
def my_int2(sweep,fs):
    int_factor = 100
    time = np.linspace(0,(sweep.size-1)/fs, sweep.size)
    f=interpolate.interp1d(time, sweep,  kind='cubic')
    t2=np.linspace(time[0], time[time.size-1], int_factor*sweep.size)    #interpolate 100x
    tmp1 = f(t2)      #upsample
    #integrate twics
    tmp_int2 = integrate.cumulative_trapezoid(integrate.cumulative_trapezoid(tmp1, initial=0),initial=0)
    f1=interpolate.interp1d(t2, tmp_int2,  kind='cubic')  #downsample
    sweep_int2 = f1(time)/(fs*fs*int_factor*int_factor)   #downsample
    return sweep_int2

#function to read in an existing sweep file - and split it onto two new sweep-files,
#each with sparse (and interleaved) frequencies
def make_sparse_sweeps(filename:str, f_width:int, desired_sweep_len:float, phase_add:int=0, amplitude_overlap:float=0.0):
    #INPUT PARAMETERS
      #filename              = string containing the full path to a nominal sweep
      #f_width                = the width (in Hz) of the frequency bands to assign to each PM.
      #                         Example: if f_width=2 then: 0-2Hz-->PM1, 2-4Hz-->PM2, 4-6Hz-->PM1 ....
      #desired_sweep_len      = how long you want tot output sweeps to be
      #phase_add              = a phase delay (in degrees) to add to the two output sweeps
      #amplitude_overlap      = how much signal outside the frequecy band of each sparse sweep that is included (this can be used to ensure a smooth NS signal)
    #OUPTUT PARAMETERS:::
      #[df1 and df2]          = two dataframes containing the sparse sweeps.
      #                         They can be saved with the command: df1.to_csv('my1.csv', sep="\t", index=False, header=False)

    #read in the tab separated sweep file
    df = pd.read_csv(filename,   header=None, delimiter=r"\s+", engine='python')
    time              = np.array(df.iloc[:,0])
    double_integral   = np.array(df.iloc[:,1])
    NS                = np.array(df.iloc[:,2])
    u_freq            = np.array(df.iloc[:,3])
    applied_phaseroll = np.array(df.iloc[:,4])
    amplitude_NS      = np.array(df.iloc[:,5])

    si = abs(time[1]-time[0])
    fs=1.0/si

    fbeg=math.floor(min(u_freq))  #need to start at integer
    fend=math.ceil(max(u_freq))   #need to stop at integer

    #print("The frequency range=[",fbeg,"-",fend,"] Hz.")

    #just some sanity checking
    if(abs(fbeg-fend) <2*f_width):
        print("The guiding signal does not contain enough frequencies to meaningfully split it up")
        return -1
    if(si != 0.00025):
        print("This file did not have the normal 0.00025 sampling rate. si=",si)
    if(si >16*0.00025):
        print("The si found was ",si," Something is wrong!???")
        return -1
    if (time[time.size-1] < 0.2):
        print("The sweep length is only ",time[time.size-1]," sec. Something is wrong!???")
        return -1
    if (time[time.size-1] > 60):
        print("The sweep length is ",time[time.size-1]," sec. Something is wrong!???")
        return -1

    #some further checks on the initial sweep...
    if(NS[NS.size-1]!=0): #apply a taper to NS
        NS=applyTaper(NS, 50)
        amplitude_NS = envelope(NS)
    if(double_integral[double_integral.size-1]!=0):
        double_integral=applyTaper(double_integral,50)
    amplitude_di   = envelope(double_integral)

    #Determening the default (minimum) length of the sweeeps (by just looking at the freq of the existing sweep and adding)
    count=0
    sweep_len1=0
    sweep_len2=0
    for f in range(fbeg, fend, f_width):
        count +=1
        if (count%2!=0):
            sweep_len1 +=si*get_nominal_sweep_length(f, min(fend,f+f_width), u_freq)  #in sec
        else:
            sweep_len2 += si*get_nominal_sweep_length(f, min(fend,f+f_width), u_freq)

    #Declare needed arrays - and generate the sparse sweeeps
    count=0   #counter
    b1=np.deg2rad(phase_add)      #to keep track of the phase
    b2=np.deg2rad(phase_add)
    f1=np.arange(0)               #to keep track of the freq
    f2=np.arange(0)
    sparse_sweep1=np.arange(0)    #the sweep to build up
    sparse_sweep2=np.arange(0)
    for f in range(fbeg, fend, f_width):
        count +=1
        if (count%2!=0):
            [sweep, b1] = generate_sweep(f, min(fend,f+f_width), fs, b1, u_freq, (1.0-amplitude_overlap)*desired_sweep_len/sweep_len1)
            sparse_sweep1= np.append(sparse_sweep1, sweep)
            f1 = np.append(f1, np.linspace(f, min(fend,f+f_width), sweep.size) )     #store the frequencies

            #it seems like the best way to avoid kinks int f'' is to also generate the freq in between - although sweeping through these much faster...
            [sweep, b2] = generate_sweep(f, min(fend,f+f_width), fs, b2, u_freq, amplitude_overlap*desired_sweep_len/sweep_len2)
            sparse_sweep2= np.append(sparse_sweep2, sweep)
            f2 = np.append(f2, np.linspace(f, min(fend,f+f_width), sweep.size) )     #store the frequencies

        else:
            [sweep, b2] = generate_sweep(f, min(fend,f+f_width), fs, b2, u_freq, (1.0-amplitude_overlap)*desired_sweep_len/sweep_len2)
            sparse_sweep2= np.append(sparse_sweep2, sweep)
            f2 = np.append(f2, np.linspace(f, min(fend,f+f_width), sweep.size) )     #store the frequencies

            #it seems like the best way to avoid kinks int f'' is to also generate the freq in between - although sweeping through these much faster...
            [sweep, b1] = generate_sweep(f, min(fend,f+f_width), fs, b1, u_freq, amplitude_overlap*desired_sweep_len/sweep_len1)
            sparse_sweep1= np.append(sparse_sweep1, sweep)
            f1 = np.append(f1, np.linspace(f, min(fend,f+f_width), sweep.size) )     #store the frequencies


    #flip everything around
    if(u_freq[0]>u_freq[u_freq.size-1]):
        sparse_sweep1 = np.flipud(sparse_sweep1)
        sparse_sweep2 = np.flipud(sparse_sweep2)
        f1=np.flipud(f1)
        f2=np.flipud(f2)
        u_freq=np.flipud(u_freq)
        amplitude_NS=np.flipud(amplitude_NS)
        amplitude_di=np.flipud(amplitude_di)


    #just double checking that they have exactly the correct (desired) length
    [sparse_sweep1, f1] = check_sweep_length(sparse_sweep1, f1, 1+desired_sweep_len*fs)
    [sparse_sweep2, f2] = check_sweep_length(sparse_sweep2, f2, 1+desired_sweep_len*fs)

    #scale the amplitudes and apply a taper at the ends
    sparse_sweep1 = scaleAmplitudes(sparse_sweep1, u_freq, amplitude_di, f1)
    sparse_sweep2 = scaleAmplitudes(sparse_sweep2, u_freq, amplitude_di, f2)


    #gentle filtering to remove DC-shift - and potential kinks
    sparse_sweep1=butter_bandpass_filter(sparse_sweep1, round(0.3*fbeg), round(1.5*fend), fs, order=4)
    sparse_sweep2=butter_bandpass_filter(sparse_sweep2, round(0.3*fbeg), round(1.5*fend), fs, order=4)

    sparse_sweep1=applyTaper(sparse_sweep1)  #tapering off does create HF noise - which cannot be avoided
    sparse_sweep2=applyTaper(sparse_sweep2)  #this will show up as a 'kink' in the f'' . If we filter it, the taper will also disappear, so that is ot a "solution"

    plotsweep(NS, si, "Plot the initial driving sweep:")
    plotsweep(double_integral, si, "Plot the double int of the initial sweep:")

    plotsweep(sparse_sweep1, si, "Sparse sweep #1 (double int):")
    plotsweep(sparse_sweep2, si, "Sparse sweep #2 (double int):")

    ns1 = my_der2(sparse_sweep1,fs)
    #we can also filter this f'', but this alters the signal in unpredictable ways, so I am not so keen on doing that...
    #ns1=butter_bandpass_filter(ns1, round(0.3*fbeg), round(1.5*fend), fs, order=4)
    ns1=applyTaper(ns1)

    ns2 = my_der2(sparse_sweep2,fs)
    #ns2_tmp=ns2
    #ns2=butter_bandpass_filter(ns2, round(0.1*fbeg), round(3.5*fend), fs, order=4)
    ns2=applyTaper(ns2)

    plotsweep(ns1, si, "The sparse sweep #1 (QC: should be smooth):", True)
    plotsweep(ns2, si, "The sparse sweep #2 (QC: should be smooth):", True)

    #plotsweep(ns2-ns2_tmp, si, "Diff ns2 #2 (after filtering):", True)

    #saving stuff for the new pm1
    time1               = np.linspace(0,desired_sweep_len, sparse_sweep1.size)
    double_integral1    = sparse_sweep1
    NS1                 = ns1   #double derivative of the one above
    u_freq1             = f1
    amplitude_phaseroll1= phase_add*np.ones(sparse_sweep1.size)
    amplitude_NS1       = envelope(NS1)
    df1=pd.DataFrame(data=[time1, double_integral1, NS1, u_freq1, amplitude_phaseroll1, amplitude_NS1], index=["time", "double_integral", "NS", "u_freq", "phaseroll", "envelope(NS)"])
    df1=df1.T

    #saving stuff for the new pm2
    time2               = np.linspace(0,desired_sweep_len, sparse_sweep1.size)
    double_integral2    = sparse_sweep2
    NS2                 = ns2     #double derivative of the one above
    u_freq2             = f2
    amplitude_phaseroll2= phase_add*np.ones(sparse_sweep2.size)
    amplitude_NS2       = envelope(NS2)

    df2=pd.DataFrame(data=[time2, double_integral2, NS2, u_freq2, amplitude_phaseroll2, amplitude_NS2], index=["time", "double_integral", "NS", "u_freq", "phaseroll", "envelope(NS)"])
    df2=df2.T

    return [df1, df2]

def make_sweeps():
    st.header("Split a nominal sweep into two sparse sweeps:")
    st.write("Explanation: Assume we have a nominal sweep file of a sweep going from 30-150Hz: We want to split the frequency content of this nominal sweep onto two new sweeps. Set the width (in Hz) of 5. This will then give you sweep #1 of 30-35Hz, 40-45Hz, 50-55Hz and so on, while sweep #2 will be 35-40Hz, 45-50Hz, 55-60Hz and so on.")
    st.write("Both the new sparse sweep files can be QC'ed and downloaded below...")
    file_name = st.file_uploader("Select a nominal sweep file (6 cols tab delimited ascii)")
    f_width=st.slider("Width (in Hz) of frequency bands",1,10,2,1)
    desired_sweep_len=st.slider("Desired sweep length (sec) of the sparse sweeps", 1.0,40.0,5.0,0.1)
    phase_add=st.slider("Phase delay added to the sparse sweeps (deg)", -360,360, 0,1)
    st.write("The frequency overlap (0-1) range. 0 means no overlap, while 1 means full overlap (the sweeps will be almost identical). With a larger overlap the NS will appear more smooth.")
    amplitude_overlap=st.slider("Frequency overlap (0-1) between the two sweeps",0.0,1.0,0.0,0.01)

    if (file_name != None):
        [df1, df2]=make_sparse_sweeps(file_name, f_width, desired_sweep_len, phase_add, 0.5*amplitude_overlap)

        df1.to_csv('sparse_sweep1.data', sep="\t", index=False, header=False)
        df2.to_csv('sparse_sweep1.data', sep="\t", index=False, header=False)

        if st.button('Download sparse_sweep1'):
            tmp_download_link = download_link(df1, 'sparse_sweep1.data', 'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

        if st.button('Download sparse_sweep2'):
            tmp_download_link = download_link(df2, 'sparse_sweep2.data', 'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)


    # elif(type=="Linear"):
    #     fbeg=st.slider("Minimum frequency",1,150,30,1)
    #     fend=st.slider("Maximum frequency",fbeg,250,150,1)
    #     s_len=0
    #     fs=4000
    #     #get the nominal sweep length first
    #     for i in range(fbeg,fend-1):
    #         s_len += math.ceil(fs/i)
    #     #then we can construct the linear underlying Frequency
    #     u_freq=[]
    #     for i in range(fbeg,fend-1):
    #         u_freq= np.append(u_freq, (np.linspace(i,i+1,round((fs*desired_sweep_len/s_len)*math.ceil(fs/i)))))
    #     plotsweep(u_freq, 1.0/fs, "the underlying freq:",False)
    #     #we then need to get the maximum allowed amplitude at each freq - start by reading this from Robeert's sweep
    #     dir_name="Vib_Database"
    #     file_hf_omni=os.path.join(dir_name,"v17_OUTPUT_DOWNSWEEP_0deg_phase_sec_disp_acc_Hz_rollrad_VIBGROUP_1_OBN_OMNI.data")
    #     df=readSignature2(file_hf_omni)
    #     u_freq_hf            = np.array(df.iloc[:,3])
    #     amplitude_NS_hf      = np.array(df.iloc[:,5])
    #     file_lf_omni=os.path.join(dir_name,"v17_OUTPUT_DOWNSWEEP_0deg_phase_sec_disp_acc_Hz_rollrad_VIBGROUP_2_OBN_OMNI.data")
    #     df=readSignature2(file_lf_omni)
    #     u_freq_lf            = np.array(df.iloc[:,3])
    #     amplitude_NS_lf      = np.array(df.iloc[:,5])
    #     time_lf              = np.array(df.iloc[:,0])
    #     double_integral_lf   = np.array(df.iloc[:,1])
    #     NS_lf               = np.array(df.iloc[:,2])
    #
    #     u_f_guide = np.concatenate(( u_freq_hf, u_freq_lf))
    #     amplitude_guide = np.concatenate(( amplitude_NS_hf, amplitude_NS_lf))
    #
    #     plotsweep(u_f_guide , 1.0/fs, "the underlying freq guide:",False)
    #     plotsweep(amplitude_guide, 1.0/fs, "the amplitude guide:",False)
    #     plotsweep(amplitude_NS_lf,  1.0/fs, "the lf amplitude part", False)
    #     plotsweep(NS_lf,  1.0/fs, "the lf signal", True)
    #     plotsweep(u_freq_lf , 1.0/fs, "freq",True )
    #     #there is something wrong about the lf part from Robert - so need to fix this...
    #
    #     ####envelope=[]
    #     #for i in range(fbeg,fend-1):
    #
    #
    #
    #     print("Work in progress!")
    # else:
    #     print("Not implemented yet")





if __name__ == "__main__":
    make_sweeps()
