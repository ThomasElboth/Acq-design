import streamlit as st
import numpy as np
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

#compute the "valid" source separation alternatives to get the desired bin-size
def get_valid_source_sep_alternatives(no_sources, streamer_sep):
    if no_sources==0:
        return [0]
    source_separation = [] #compute the valid source separatins for non-overlapping bins
    for i in range(0,100,no_sources):
        source_separation.append(round(100*(streamer_sep/no_sources)*(i+1))/100)
        if(len(source_separation)>5):
            break
    source_separation.append(0)
    #Also add a number of no-god options for the 'stupid' user
    for i in range(round(source_separation[0])+1, round(source_separation[1])-1):
        source_separation.append(i)
    for i in range(10, round(source_separation[0])-1):
        source_separation.append(i)

    return source_separation


#script that gets the acq parameters for the node survey - and returns the src andreceiver locations
def get_source_and_rec_locations_nodes(survey_area_x, survey_area_y, node_dx, node_dy, source_sep, spi,no_sources, shooting_area_overlap, use_dipole_source, no_saillines):
    survey_area_x = survey_area_x*1000 #in m
    survey_area_y = survey_area_y*1000 #in m
    shooting_area_overlap = shooting_area_overlap*1000 #in m

    #Create numpy arrays wth all receiver_x (Rx) and receiver_y (Ry) positons
    Rx = np.linspace(0, survey_area_x, 1+round(survey_area_x/node_dx))
    Ry = np.linspace(0, survey_area_y, 1+round(survey_area_y/node_dy))
    #and all Sx and Sy (source positions)
    Sx = np.linspace(-shooting_area_overlap, survey_area_x+shooting_area_overlap, 1+round((survey_area_x+2*shooting_area_overlap)/(spi*no_sources)))
    Sy = np.linspace(-shooting_area_overlap, survey_area_y+shooting_area_overlap, 1+round((survey_area_y+2*shooting_area_overlap)/(source_sep)))

    fig = go.Figure()
    #make and plot all the recievers
    R=np.zeros((len(Rx)*len(Ry),2))
    count=0
    for i in range(0,len(Rx)):
        for j in range(0,len(Ry)):
            R[count,0] = Rx[i]
            R[count,1] = Ry[j]
            count=count+1


    #if we have a dipole source - we drop saillines 1--4--7--10--13....
    source_lines =  np.ones(len(Sy))
    count2=0
    while (count2<len(Sy)):
        for i in range(0,no_sources):  #lines to be shot
            if(count2<len(Sy)):
                source_lines[count2]=1
            count2=count2+1
        for i in range(0,2*no_sources): #lines to be interpolated
            if((count2<len(Sy)) and (use_dipole_source==True)):
                source_lines[count2]=0
            count2=count2+1

    S=np.zeros(( (len(Sx)*len(Sy)),2 ))
    count1=0
    for i in range(0,len(Sx)):
        for j in range(0,len(Sy)):
            if(source_lines[j]!=0):
                S[count1,0] = Sx[i]
                S[count1,1] = Sy[j]
                count1=count1+1

    #drop all entries in the S(ource_point) matrix that are zero
    S = S[[i for i, x in enumerate(S) if x.any()]]

    return [R, S]  #returning the S(ource) and R(eciever) locations

#Plot the survey area layout
def plot_node_survey_layout(R, S, S_dipole):

    #plotting both without and with depole source
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=R[:,0], y=R[:,1], mode='markers', name='node', marker_size=3 ))
    fig.add_trace(go.Scatter(x=S[:,0], y=S[:,1], mode='markers', name='source point', marker_size=1))
    survey_area = (max(R[:,0])-min(R[:,0])) * (max(R[:,1])-min(R[:,1]))/1000000  #in km2
    no_sp = sum(S[:,0].shape)
    no_nodes =sum(R[:,0].shape)
    txt = ('Node survey of %d km2 with %d nodes and %d shotpoints.' %(survey_area, no_nodes, no_sp))
    fig.update_layout(title=txt,  xaxis_title='Meter',yaxis_title='Meter')

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=R[:,0], y=R[:,1], mode='markers', name='node', marker_size=3 ))
    fig1.add_trace(go.Scatter(x=S_dipole[:,0], y=S_dipole[:,1], mode='markers', name='source point (dipole)', marker_size=1))
    survey_area = (max(R[:,0])-min(R[:,0])) * (max(R[:,1])-min(R[:,1]))/1000000  #in km2
    no_sp = sum(S_dipole[:,0].shape)
    no_nodes =sum(R[:,0].shape)
    txt = ('Node survey of %d km2 with %d nodes and %d shotpoints (dipole source).' %(survey_area, no_nodes, no_sp))
    fig1.update_layout(title=txt,  xaxis_title='Meter',yaxis_title='Meter')

    return [fig, fig1]

def get_offset_from_one_shot_on_nodes(R, S, shooting_area_overlap):

    #start by finding a source in the middle of the survey_area
    node_area_x = max(R[:,0])
    node_area_y = max(R[:,1])
    survey_area_x = node_area_x + 2*shooting_area_overlap*1000
    survey_area_y = node_area_y + 2*shooting_area_overlap*1000
    source_dx = max(np.diff(S[:,0])) #trick to get the node DX in one line
    source_dy =  S[1,1]-S[0,1]

    node_dx = max(np.diff(R[:,0])) #trick to get the node DX in one line
    node_dy = R[1,1]-R[0,1]

    Sx = np.linspace(0, survey_area_x, 1+round(survey_area_x/source_dx))
    Sy = np.linspace(0, survey_area_y, 1+round(survey_area_y/source_dy))
    my_source_x = Sx[round(len(Sx)/2)]
    my_source_y = Sy[round(len(Sy)/2)]

    offsets = np.zeros((len(R)))
    cmp_x = np.zeros((len(R)))
    cmp_y = np.zeros((len(R)))
    for i in range(0, len(R)): #for all receivers
        offsets[i] = 0.5*math.sqrt( (my_source_x-R[i,0])**2 + (my_source_y-R[i,1])**2  )
        cmp_x[i] = 0.5*(my_source_x+R[i,0]) #THIS IS ONLY APPROX FOR DEEP REFLECTIONS (node sits on the WB)
        cmp_y[i] = 0.5*(my_source_y+R[i,1]) #THIS IS ONLY APPROX FOR DEEP REFLECTIONS (node sits on the WB)

    return [offsets, cmp_x, cmp_y, node_dx, node_dy, node_area_x, node_area_y, survey_area_x, survey_area_y]

def plot_node_survey_offset_distribution_from_shot(R, S, S_dipole, spi, no_sources,shooting_area_overlap):

    [offsets, cmp_x, cmp_y, node_dx, node_dy, node_area_x, node_area_y, survey_area_x, survey_area_y] = get_offset_from_one_shot_on_nodes(R, S, shooting_area_overlap)

    #Make a offset histogram of the offsets 0 2*SPI (flip to flip)
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=offsets,
    xbins=dict( start=0, end=4000, size=round(2*spi*no_sources)  ),
    name='',
    opacity=0.5,
    marker_color='blue'
    ))

    fig.update_layout(
        title_text='Single shot offset distribution on '+str(len(R))+' nodes with spacing (dx, dy)=<br>('+str(round(node_dx))+', '+str(round(node_dy))+') m. The shooting area: '+str(round(survey_area_x/1000))+' X '+str(round(survey_area_y/1000))+' km.',
        xaxis_title_text='Offset groups of '+str(round(2*spi*no_sources))+' m.', # xaxis label
        yaxis_title_text='TraceCount', # yaxis label
        bargap=0.1, # gap between bars of adjacent location coordinates
        )

    #st.plotly_chart(fig,use_container_width=False)
    return fig

#Plot the offset distribution from one single shot - as recorded on all receivers
def plot_node_survey_offset_distribution_on_one_receiver(R, S, S_dipole, spi, no_sources):

    #start by finding the receiver (node) in the middle of the survey_area
    survey_area_x = max(R[:,0])
    survey_area_y = max(R[:,1])
    node_dx = max(np.diff(R[:,0])) #trick to get the node DX in one line
    source_dy =  S[1,1]-S[0,1]
    node_dy = R[1,1]-R[0,1]

    Rx = np.linspace(0, survey_area_x, 1+round(survey_area_x/node_dx))
    Ry = np.linspace(0, survey_area_y, 1+round(survey_area_y/node_dy))
    my_node_x = Rx[round(len(Rx)/2)]
    my_node_y = Ry[round(len(Ry)/2)]

    offsets = np.zeros((len(S)))
    for i in range(0, len(S)): #for all shots
        offsets[i] = 0.5*math.sqrt( (my_node_x-S[i,0])**2 + (my_node_y-S[i,1])**2  )
    offsets= np.sort(offsets)

    offsets_dipole = np.zeros((len(S)))
    for i in range(0, len(S_dipole)): #for all shots
        offsets_dipole[i] = 0.5*math.sqrt( (my_node_x-S_dipole[i,0])**2 + (my_node_y-S_dipole[i,1])**2  )
    offsets_dipole= np.sort(offsets_dipole)
    offsets_dipole = offsets_dipole[[i for i, x in enumerate(offsets_dipole) if x.any()]] #remove all zero-values:

    #Make a offset histogram of the offsets 0 2*SPI (flip to flip)
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=offsets,
    xbins=dict( start=0, end=4000, size=round(no_sources*spi*2)  ),
    name='normal_source',
    opacity=0.5,
    marker_color='blue'
    ))

    fig.add_trace(go.Histogram(x=offsets_dipole,
    xbins=dict( start=0, end=4000, size=round(no_sources*spi*2)  ),
    name='dipole_source',
    opacity=0.5,
    marker_color='red'
    ))

    fig.update_layout(
        title_text='Singel node CMP offset distribution for '+str(len(S))+' shots. SPI(flip-to-flop)='+str(spi)+'m<br>with '+str(no_sources)+' sources. The source-line sep is '+str(source_dy)+' m.', # title of plot
        xaxis_title_text='Offset groups of '+str(round(2*spi*no_sources))+' m.', # xaxis label
        yaxis_title_text='TraceCount', # yaxis label
        bargap=0.1, # gap between bars of adjacent location coordinates
        )

    #st.plotly_chart(fig,use_container_width=False)
    return fig

def get_source_y(no_sources, mid, source_sep):
    source_y = np.zeros((no_sources))
    if(no_sources==1):
        source_y[0] = mid
    elif (no_sources==2):
        source_y[0] = mid-0.5*source_sep
        source_y[1] = mid+0.5*source_sep
    elif(no_sources==3):
        source_y[0] = mid-1.0*source_sep
        source_y[1] = mid
        source_y[2] = mid+1.0*source_sep
    elif(no_sources==4):
        source_y[0] = mid-1.5*source_sep
        source_y[1] = mid-0.5*source_sep
        source_y[2] = mid+0.5*source_sep
        source_y[3] = mid+1.5*source_sep
    elif(no_sources==5):
        source_y[0] = mid-2*source_sep
        source_y[1] = mid-1*source_sep
        source_y[2] = mid
        source_y[3] = mid+1*source_sep
        source_y[4] = mid+2*source_sep
    elif(no_sources==6):
        source_y[0] = mid-2.5*source_sep
        source_y[1] = mid-1.5*source_sep
        source_y[2] = mid-0.5*source_sep
        source_y[3] = mid+0.5*source_sep
        source_y[4] = mid+1.5*source_sep
        source_y[5] = mid+2.5*source_sep
    elif( (no_sources>6) and (no_sources % 2)==0): #even
        tmp = -(no_sources/2)+0.5
        for i in range(0, no_sources):
            source_y[i] = mid + (tmp+i)*source_sep
    elif ( (no_sources>6) and (no_sources % 2)!=0): #odd
        tmp = -math.floor(no_sources/2)
        for i in range(0, no_sources):
            source_y[i] = mid + (tmp+i)*source_sep
    else:
        st.write('you should not come here!!!')

    return source_y

#Compute the source and streamer layout - and return the appropriate arrays
def get_source_and_streamer_layout(no_lf_sources, source_lf_sep, no_hf_sources, source_hf_sep, no_streamers, streamer_sep, streamer_len, layback, place_lf_in_center):

    #first to the streamer layout: x=inline, y=crossline
    streamer_y = np.linspace(0, streamer_sep*(no_streamers-1), no_streamers)
    streamer_x = np.linspace(layback,layback+streamer_len, 1+round(streamer_len/12.5))
    mid = streamer_sep*(no_streamers-1)/2 #the center of the spread - right behind the boat

    source_lf_x = []
    source_lf_y = []
    source_hf_x = []
    source_hf_y = []

    #then we start on the sources
    if place_lf_in_center==True: #we place the lf-soures in the center - which is good both geophysically and practically (they are deep)
        source_lf_y=get_source_y(no_lf_sources, mid, source_lf_sep) #this is the default
        #get the next up separation instead
        if no_hf_sources>=2: #always even due to symmetry
            source_hf_y.append(mid-0.5*source_hf_sep)
            source_hf_y.append(mid+0.5*source_hf_sep)
        if no_hf_sources>=4:
            source_hf_y.append(mid-1.5*source_hf_sep)
            source_hf_y.append(mid+1.5*source_hf_sep)
        if no_hf_sources>=6:
            source_hf_y.append(mid-2.5*source_hf_sep)
            source_hf_y.append(mid+2.5*source_hf_sep)
        if no_hf_sources>=8:
            source_hf_y.append(mid-3.5*source_hf_sep)
            source_hf_y.append(mid+3.5*source_hf_sep)
        if no_hf_sources>8:
            st.write('Only up to 8 sources of each type is supported for now.')

    else:
        source_hf_y=get_source_y(no_hf_sources, mid, source_hf_sep) #when/if using  gradient source in the middle of the spread
        if no_lf_sources>=2:#always even due to symmetry
            source_lf_y.append(mid-0.5*source_lf_sep)
            source_lf_y.append(mid+0.5*source_lf_sep)
        if no_lf_sources>=4:
            source_lf_y.append(mid-1.5*source_lf_sep)
            source_lf_y.append(mid+1.5*source_lf_sep)
        if no_lf_sources>=6:
            source_lf_y.append(mid-2.5*source_lf_sep)
            source_lf_y.append(mid+2.5*source_lf_sep)
        if no_lf_sources>=8:
            source_lf_y.append(mid-3.5*source_lf_sep)
            source_lf_y.append(mid+3.5*source_lf_sep)
        if no_lf_sources>8:
            st.write('Only up to 8 sources of each type is supported for now.')

    source_lf_x=np.zeros((len(source_lf_y)))
    source_hf_x=np.zeros((len(source_hf_y)))

    return [source_lf_x, source_lf_y, source_hf_x, source_hf_y, streamer_x, streamer_y]


#Compute the source and streamer layout - and return the appropriate arrays
def get_source_and_streamer_layout_short(no_sources, source_sep, no_streamers, streamer_sep, streamer_len, layback):

    print(no_sources)
    streamer_x = np.linspace(0, streamer_sep*(no_streamers-1), no_streamers)
    streamer_y = np.linspace(layback,layback+streamer_len, 1+round(streamer_len/12.5))

    source_x = np.zeros((no_sources))
    source_y = np.zeros((no_sources))
    mid = streamer_sep*(no_streamers-1)/2
    if(no_sources==1):
        source_x[0] = mid
    elif (no_sources==2):
        source_x[0] = mid-0.5*source_sep
        source_x[1] = mid+0.5*source_sep
    elif(no_sources==3):
        source_x[0] = mid-source_sep
        source_x[1] = mid
        source_x[2] = mid+source_sep
    elif(no_sources==4):
        source_x[0] = mid-1.5*source_sep
        source_x[1] = mid-0.5*source_sep
        source_x[2] = mid+0.5*source_sep
        source_x[3] = mid+1.5*source_sep
    elif(no_sources==5):
        source_x[0] = mid-2*source_sep
        source_x[1] = mid-1*source_sep
        source_x[2] = mid
        source_x[3] = mid+1*source_sep
        source_x[4] = mid+2*source_sep
    elif(no_sources==6):
        source_x[0] = mid-2.5*source_sep
        source_x[1] = mid-1.5*source_sep
        source_x[2] = mid-0.5*source_sep
        source_x[3] = mid+0.5*source_sep
        source_x[4] = mid+1.5*source_sep
        source_x[5] = mid+2.5*source_sep

    elif( (no_sources>6) and (no_sources % 2)==0): #even
        print("No sources:", no_sources)
        tmp = -(no_sources/2)+0.5
        for i in range(0, no_sources):
            source_x[i] = mid + (tmp+i)*source_sep
    elif ( (no_sources>6) and (no_sources % 2)!=0): #odd
        tmp = -math.floor(no_sources/2)
        for i in range(0, no_sources):
            source_x[i] = mid + (tmp+i)*source_sep

    else:
        st.write('You should not come here - contact programming!')

    return [source_x, source_y, streamer_x, streamer_y]
#compute the offsets and CMP's from one shot on a seismic spread
def get_offsets_from_one_shot_on_a_spread(source_x, source_y, streamer_x, streamer_y):

    #We set the source_pos as (0,0) central source in case of 3 or 5 sources
    source_x=0
    source_y=0

    #Then loop over all receivers - to get the offsets
    offsets = np.zeros((len(streamer_x)*len(streamer_y)))
    cmp_x = np.zeros((len(streamer_x)*len(streamer_y)))
    cmp_y = np.zeros((len(streamer_x)*len(streamer_y)))

    count=0
    for i in range(0, len(streamer_x)):
        for j in range(0, len(streamer_y)):
            offsets[count] = 0.5*( math.sqrt(streamer_x[i]**2 + streamer_y[j]**2) )
            cmp_x[count] = 0.5*streamer_x[i]
            cmp_y[count] = 0.5*streamer_y[j]
            count=count+1

    return [offsets, cmp_x, cmp_y]

def plot_compare_streamer_and_node_offset(no_sources, source_sep, spi, no_streamers, streamer_sep, streamer_len, layback, R, S, S_dipole, shooting_area_overlap ):

    [source_x, source_y, streamer_x, streamer_y] = get_source_and_streamer_layout_short(no_sources, source_sep, no_streamers, streamer_sep, streamer_len, layback)
    #get the offsets from the node-survey
    [offsets_streamer, cmp_x_streamer, cmp_y_streamer]=get_offsets_from_one_shot_on_a_spread(source_x, source_y, streamer_x, streamer_y)
    #get the offsets from the streamer survey
    [offsets_node, cmp_x_node, cmp_y_node, node_dx, node_dy, node_area_x, node_area_y, survey_area_x, survey_area_y] = get_offset_from_one_shot_on_nodes(R, S, shooting_area_overlap)

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=offsets_streamer,
    xbins=dict( start=0, end=4000, size=round(no_sources*spi*2)  ),
    name='Streamers',
    opacity=0.5,
    marker_color='blue'
    ))

    fig.add_trace(go.Histogram(x=offsets_node,
    xbins=dict( start=0, end=4000, size=round(no_sources*spi*2)  ),
    name='Nodes',
    opacity=0.5,
    marker_color='green'
    ))

    txt=""
    if(layback < 0):
        txt="(Topseis mode)"

    fig.update_layout(
        title_text='Single shot (middle of the spread) offset distribution on:<br>'+str(no_streamers)+' str@'+str(streamer_sep)+'m and length '+str(streamer_len)+'m. Source layback: '+str(layback)+'m. '+txt+'<br>'+str(len(R))+' nodes with spacing (dx, dy)=('+str(round(node_dx))+', '+str(round(node_dy))+')m. The node area: '+str(round(node_area_x/1000))+' X '+str(round(node_area_x/1000))+' km.',
        xaxis_title_text='Offset groups of '+str(round(2*spi*no_sources))+' m.', # xaxis label
        yaxis_title_text='TraceCount', # yaxis label
        bargap=0.1, # gap between bars of adjacent location coordinates
        )

    #st.plotly_chart(fig,use_container_width=False)
    return fig

def plot_offset_classes(streamer_sep, source_x, source_y, streamer_x, streamer_y, offset_class_size, no_hf_sources_on_vessel1):
    fig = go.Figure()
    no_streamers = len(streamer_x)
    no_sources = len(source_x)
    source_sep = 0.0
    try:
        source_sep = abs(source_x[0] - source_x[1])
        source_sep = "{:3.2f}".format(source_sep)
    except:
        source_sep = 0


    dy = streamer_sep/(2*no_hf_sources_on_vessel1)

    no_channels = len(streamer_y)
    my_colors = ['red','green','blue','purple','cyan','orange','black','red','green','blue','purple','cyan','orange','black']

    no_offset_classes=10
    #these offset groups are for 'normal' towing (off-end)
    offset_groups = np.linspace(0,no_offset_classes*offset_class_size, no_offset_classes+1) #make the offset classes

    for i in range(0, no_sources):
        for j in range(0, no_streamers):
            tmp=(source_x[i]-streamer_x[j])**2
            cmp_x= 0.5*(source_x[i] + streamer_x[j])
            for k in range(0, no_channels):
                offset = math.sqrt(tmp +streamer_y[k]**2)
                if (offset > no_offset_classes*offset_class_size) and (streamer_y[0] > 0) : #avoid unnessesary computations
                    break

                group = binarySearch(offset_groups, offset)
                if group <10:
                    fig.add_trace(go.Scatter(x=[cmp_x-0.48*dy, cmp_x+0.48*dy],y=[group*offset_class_size, group*offset_class_size], line=dict(color=my_colors[i], width=7), mode='lines' ))
                    if(streamer_y[0] < 0): #topseis mode
                        fig.add_trace(go.Scatter(x=[cmp_x-0.48*dy, cmp_x+0.48*dy],y=[-1*group*offset_class_size, -1*group*offset_class_size], line=dict(color=my_colors[i], width=7), mode='lines' ))

        #add proper y-ticks
        fig.update_layout(yaxis = dict(tick0=0, dtick=offset_class_size))
        fig.update_yaxes(range=[0,no_offset_classes*offset_class_size])
        if(streamer_y[0] < 0): #topseis mode
                fig.update_yaxes(range=[-no_offset_classes*offset_class_size,no_offset_classes*offset_class_size])
        txt='Offset distribution for '+str(no_streamers)+' streamers@'+str(streamer_sep)+'m and '+str(no_hf_sources_on_vessel1)+' sources@'+str(source_sep)+'m.<br>The offset class size is '+str(offset_class_size)+'m.'
        fig.update_layout(title=txt, xaxis_title='Crossline (Meter)',yaxis_title='CMP offset class (Meter)', showlegend=False)

    return fig

def draw_vessel_outline(fig, mid, offset_inline, offset_crossline):
    mid=mid + offset_crossline
    fig.add_trace(go.Scatter(x=[mid-12, mid+12], y=[-300+offset_inline, -300+offset_inline],  line=dict(color='black', width=3), mode='lines' ))
    fig.add_trace(go.Scatter(x=[mid-12, mid-12], y=[-300+offset_inline, -370+offset_inline],  line=dict(color='black', width=3), mode='lines' ))
    fig.add_trace(go.Scatter(x=[mid+12, mid+12], y=[-300+offset_inline, -370+offset_inline],  line=dict(color='black', width=3), mode='lines' ))
    fig.add_trace(go.Scatter(x=[mid-12, mid],    y=[-370+offset_inline, -400+offset_inline],  line=dict(color='black', width=3), mode='lines' ))
    fig.add_trace(go.Scatter(x=[mid+12, mid],    y=[-370+offset_inline, -400+offset_inline],  line=dict(color='black', width=3), mode='lines' ))

    return fig


#plot the layout of the source and streamer setup - birdseye-view
def plot_layout_birdseye(streamer_sep, source_x,  source_y, streamer_x, streamer_y, hf_source, no_source_vessels, source_vessel_offset_crossline, source_vessel_offset_inline):
    fig = go.Figure()
    no_streamers = len(streamer_x)
    no_sources = len(source_x)
    source_sep = 0.0
    try:
        source_sep = abs(source_x[0] - source_x[1])
        source_sep = "{:3.2f}".format(source_sep)
    except:
        print('Only one source?')

    #Compute all the CMP's to display at the bottom of the image
    cmp_x = []
    cmp_y = []
    try:
        mid=round(abs(0.5*(streamer_x[-1]-streamer_x[0])))
    except:
        st.write("No streamers - so this does not work...")
        return fig

    count=0
    for j in range(0, no_streamers):
        for i in range(0, no_sources):
            cmp_x.append(0.5*(source_x[i]+streamer_x[j]))
            cmp_x.append(0.5*(source_x[i]+streamer_x[j]))
            cmp_y.append(0.5*(source_y[i]+streamer_y[0]))   #first point (source + front streamer /2)
            cmp_y.append(0.5*(source_y[i]+streamer_y[-1]))   #last point  (source + back streamer) /2

    #make a color array - to see easily identify the different bins
    my_colors = ['red','green','blue','purple','cyan','orange','black']
    my_colors0 = []
    while (len(my_colors0)<len(cmp_x)):
        my_colors0.append(my_colors[0:no_sources])
    my_colors = [item for sublist in my_colors0 for item in sublist] #flatten the list

    my_colors2 = ['rgba(255,0,0,0.3)', 'rgba(0,255,0,0.3)', 'rgba(0,0,255,0.3)', 'rgba(128,0,128,0.3)','rgba(0,255,255,0.3)', 'rgba(255,127,80,0.3)', 'rgba(0,0,0,0.3)' ]
    my_colors02 = []
    while (len(my_colors02)<len(cmp_x)):
        my_colors02.append(my_colors2[0:no_sources])
    my_colors2 = [item for sublist in my_colors02 for item in sublist] #flatten the list

    my_dy = (cmp_x[2]-cmp_x[0])/2
    if my_dy > 100: #one source case
        my_dy = 0.5*streamer_sep

    for i in range(0, len(cmp_x)-1,2): #the cmp's (Need to go first due to the filling option)
        if(i<len(cmp_x)-2):
            my_dy = min(  (cmp_x[2]-cmp_x[0])/2, (cmp_x[i+2]-cmp_x[i])/2 )
            if(my_dy<=0):  #hack to avoid zeros - which gave wrong colors
                my_dy = abs((cmp_x[i]-cmp_x[i-2])/2)
            if my_dy > 100:  #taking care of the one source case
                my_dy = 0.5*streamer_sep
        print(my_dy)
        fig.add_shape(type="rect",
            x0=cmp_x[i]-0.9*my_dy,   y0=cmp_y[i],
            x1=cmp_x[i+1]+0.9*my_dy, y1=cmp_y[i+1],
            line=dict(
                color=my_colors[round(i/2)],
                width=0.1,
                ),
            fillcolor=my_colors2[round(i/2)],
        )

    #the sources
    no_lf_sources=0
    my_marker_size = 7*np.ones((no_sources))
    for i in range(0, no_sources): #the sources
        if hf_source[i]==False:
            no_lf_sources = no_lf_sources +1
            my_marker_size[i]=10

    fig.add_trace(go.Scatter(x=source_x, y=source_y, mode='markers', name='sources', marker_size=my_marker_size, marker_symbol='star', marker=dict(color='black', line=dict(color=my_colors, width=1))  ))

    for i in range(0, no_streamers): #the streamers
        fig.add_trace(go.Scatter(x=[streamer_x[i],streamer_x[i]], y=[streamer_y[0], streamer_y[-1]], line=dict(color='yellow', width=3), mode='lines' ))

    #draw a vessel
    for i in range(0, no_source_vessels):
        fig = draw_vessel_outline(fig, mid, i*1000*source_vessel_offset_inline, i*source_vessel_offset_crossline)

    #draw a spline from the (first) vessel to the sources
    start_x=np.zeros((no_sources))
    MP     =np.zeros((no_sources))
    for i in range(0, no_sources):
        if(i < no_sources/no_source_vessels and source_y[i]==0):
            start_x[i]=mid  #where the source_line leavs the back of the vessel
            MP[i] = (0.5*(source_x[i]+start_x[i])) -  0.2*((0.5*(source_x[i]+start_x[i])) -mid)  #param for a good lookingspline
            fig.add_trace(go.Scatter(x=[source_x[i], MP[i] , start_x[i]], y=[0,-150,-300],  line_shape='spline', line_color='gray', line_width=1, mode='lines'))

    #draw a stapled thin line from the vessel to the front of each streamer
    if streamer_y[0]>0: #Normal mode
        start_x = np.linspace( round(mid-12), round(mid+12), no_streamers )
        MP     =np.zeros((no_streamers))
        for i in range(0, no_streamers):
            MP[i] = (0.5*(streamer_x[i]+start_x[i])) -  0.4*((0.5*(streamer_x[i]+start_x[i])) -mid)  #param for a good lookingspline
            fig.add_trace(go.Scatter(x=[streamer_x[i], MP[i] , start_x[i]], y=[streamer_y[0], 0.5*(streamer_y[0]-300),-300 ],  line_shape='spline', line_color='gray', line_width=0.2, mode='lines'))

    else: #we're in Topseis mode - so draw an extra vessel up front
        fig = draw_vessel_outline(fig, mid, streamer_y[0], 0)

        start_x = np.linspace( round(mid-12), round(mid+12), no_streamers )
        MP     =np.zeros((no_streamers))
        for i in range(0, no_streamers):
            MP[i] = (0.5*(streamer_x[i]+start_x[i])) -  0.4*((0.5*(streamer_x[i]+start_x[i])) -mid)  #param for a good lookingspline
            fig.add_trace(go.Scatter(x=[streamer_x[i], MP[i] , start_x[i]], y=[streamer_y[0], 0.5*(streamer_y[0]-300+streamer_y[0]),-300+streamer_y[0] ],  line_shape='spline', line_color='gray', line_width=0.2, mode='lines'))

    fig.update_yaxes(range=[min(-450, min(source_y)), max(500, 600+max(source_y))])
    fig.update_xaxes(range=[streamer_x[0]-50, max(streamer_x[-1]+50, max(source_x)+50)])


    return fig

#plot the source and streamer layout, inline
def plot_layout_(streamer_sep, source_x, source_y, streamer_x, streamer_y, text, no_hf_sources_on_vessel1, no_source_vessels=1, fig=0, shift=0.0, make_thick=1, extra_sailline_spacing=0, forced_dy=0):

    no_streamers = len(streamer_x)
    no_sources = len(source_x)
    source_sep = 0.0
    try:
        source_sep = abs(source_x[0] - source_x[1])
        source_sep = "{:3.2f}".format(source_sep)
    except:
        source_sep = 0

    #Compute all the CMP's to display at the bottom of the image
    cmp_x = []
    cmp_y = []
    source_line_x=np.zeros((no_sources*no_streamers))
    cmp_line_x= np.zeros((no_sources*no_streamers))
    str_line_x= np.zeros((no_sources*no_streamers))

    count=0
    for j in range(0, no_streamers):
        for i in range(0, no_sources):

            cmp_x.append(0.5*(source_x[i]+streamer_x[j]))
            cmp_y.append(-1)

            source_line_x[count] = source_x[i]
            cmp_line_x[count] = cmp_x[count]
            str_line_x[count] = streamer_x[j]
            count=count+1

    #make a color array - to see easily identify the different bins
    my_colors = ['red','green','blue','purple','cyan','orange','black']
    my_colors0 = []
    while (len(my_colors0)<len(cmp_x)):
        my_colors0.append(my_colors[0:no_sources])
    my_colors = [item for sublist in my_colors0 for item in sublist] #flatten the list

    #starting the plotting - of one sailline
    if fig==0:
        fig = go.Figure()

    fig.add_trace(go.Scatter(x=source_x, y=0.08*np.ones((no_sources))+shift, mode='markers', name='sources', marker_size=round(7*make_thick), marker_symbol='star', marker=dict(color='black', line=dict(color=my_colors, width=1))  ))
    fig.add_trace(go.Scatter(x=streamer_x, y=np.zeros((no_streamers)), mode='markers', name='receivers', marker_size=9, marker=dict(color='orange', line=dict(color='yellow', width=2)) ))

    if no_sources>0:
        try:
            dy = 0.5*streamer_sep/no_hf_sources_on_vessel1
        except:
            dy = 0.5*streamer_sep/1 # just assume one source
        y1=-0.8
        if forced_dy>0: #hack to fill out the dy-width in cases where it is wanted
            dy = forced_dy

        for i in range(0, count):
            #if (abs(cmp_line_x[i]-cmp_line_x[max(0,i-1)]))>100:  #hack - since I do ot know which osurce...
            #    dy = 0.5*streamer_sep
            fig.add_trace(go.Scatter(x=[source_line_x[i],cmp_line_x[i]], y=[0.08+shift,y1+shift], line=dict(color='gray', width=0.3, dash='dot'), mode='lines' ))
            fig.add_trace(go.Scatter(x=[cmp_line_x[i], str_line_x[i]], y=[y1+shift,0], line=dict(color='gray', width=0.3, dash='dot'), mode='lines' ))
            fig.add_trace(go.Scatter(x=[cmp_line_x[i]-0.48*dy, cmp_line_x[i]+0.48*dy], y=[y1+shift,y1+shift], line=dict(color=my_colors[i], width=round(7*make_thick)), mode='lines' ))

    if no_sources>0:
        txt = ('Inline view: <br>'+str(no_streamers)+' str @'+str(streamer_sep)+' m and '+str(round(no_sources/no_source_vessels)) +' src @'+str(source_sep)+' m. The natural bin size (dy)= '+str(dy)+' m.')
    else:
        txt = ('Inline view: <br>'+str(no_streamers)+' str @'+str(streamer_sep)+' m and '+str(round(no_sources/no_source_vessels))+' src @'+str(source_sep)+' m. Effective CMP-width is '+str(0.5*no_streamers*streamer_sep)+' m.')

    fig.update_layout(title=txt,  xaxis_title='Meter', showlegend=False)
    fig.update_yaxes(visible=False, fixedrange=True)
    #st.plotly_chart(fig,use_container_width=False)

    #Also make a plot with two sail-lines to illustrate how they overlap
    shift_x = [0, extra_sailline_spacing+0.5*no_streamers*streamer_sep, 1.0*no_streamers*streamer_sep, 1.5*no_streamers*streamer_sep]
    shift_y = [0, -0.05, 0, -0.05]
    fig1 = go.Figure()
    for k in range(0,2):  #Two plots side by side
        fig1.add_trace(go.Scatter(x=source_x+shift_x[k]*np.ones((len(source_x))),   y=0.08*np.ones((no_sources))+shift_y[k],   mode='markers', name='sources', marker_size=8, marker_symbol='star', marker=dict(color='black', line=dict(color=my_colors, width=1))  ))
        fig1.add_trace(go.Scatter(x=streamer_x+shift_x[k], y=np.zeros((no_streamers))+shift_y[k], mode='markers', name='receivers', marker_size=7, marker=dict(color='orange', line=dict(color='yellow', width=2)) ))

        for i in range(0, count):
            fig1.add_trace(go.Scatter(x=[source_line_x[i]+shift_x[k],cmp_line_x[i]+shift_x[k]], y=[0.08+shift_y[k],y1+shift_y[k]], line=dict(color='gray', width=0.3, dash='dot'), mode='lines' ))
            fig1.add_trace(go.Scatter(x=[cmp_line_x[i]+shift_x[k], str_line_x[i]+shift_x[k]], y=[y1+shift_y[k],0+shift_y[k]], line=dict(color='gray', width=0.3, dash='dot'), mode='lines' ))
            fig1.add_trace(go.Scatter(x=[cmp_line_x[i]-0.48*dy+shift_x[k], cmp_line_x[i]+0.48*dy+shift_x[k]], y=[y1+shift_y[k],y1+shift_y[k]], line=dict(color=my_colors[i], width=5), mode='lines' ))

    fig1.update_layout(title="Two saillines<br>"+txt,  xaxis_title='Meter', showlegend=False)
    fig1.update_yaxes(visible=False, fixedrange=True)
    # #st.plotly_chart(fig1,use_container_width=False)
    return [fig, fig1]

def plot_combined_layout_(streamer_sep, source_lf_x, source_lf_y, source_hf_x, source_hf_y, streamer_x, streamer_y, no_source_vessels, no_hf_sources_on_vessel1):

    #first we plot the HF source - returning a fig
    fig = go.Figure()
    [fig, fig1] = plot_layout_(streamer_sep, source_hf_x, source_hf_y, streamer_x, streamer_y,  "HF source layout",no_hf_sources_on_vessel1, no_source_vessels)
    [fig, fig1] = plot_layout_(streamer_sep, source_lf_x, source_lf_y, streamer_x, streamer_y,  "LF source layout",no_hf_sources_on_vessel1, no_source_vessels,fig, shift=-0.05, make_thick=1.8)
    source_sep_hf = 0.0
    if len(source_hf_x) >1:
        source_sep_hf  = abs(source_hf_x[1]-source_hf_x[0])
    no_streamers = len(streamer_x)
    source_sep_lf =0.0
    if len(source_lf_x) >1:
        source_sep_lf  = abs(source_lf_x[1]-source_lf_x[0])

    no_hf_sources = len(source_hf_x)
    if no_hf_sources>0:
        bin_hf = "{:3.2f}".format(no_source_vessels*0.5*streamer_sep/no_hf_sources)
    else:
        bin_hf = 0.0
    if len(source_hf_x) >0 and len(source_lf_x)>0:
        txt = ('Inline view: <b>LF</b> and HF vibes:<br>'+str(no_streamers)+' str @'+str(streamer_sep)+'m. '+str(round(len(source_hf_x)/no_source_vessels))+' HF vib groups @'+str(round(source_sep_hf*100)/100)+'m and '+str(round(len(source_lf_x)/no_source_vessels))+' LF vib groups @'+str(round(source_sep_lf*100)/100)+'m. <br>X-line bin sizes (HF, LF)=('+str(bin_hf)+', '+str(no_source_vessels*0.5*streamer_sep/len(source_lf_x))+')m. Sailline width='+str(0.5*no_streamers*streamer_sep)+'m.')
    else:
        txt = ('Inline view: <b>LF</b> and HF vibes:<br>'+str(no_streamers)+' str @'+str(streamer_sep)+'m. '+str(len(source_hf_x))+' HF vib groups @'+str(round(source_sep_hf*100)/100)+'m and '+str(len(source_lf_x))+' LF vib groups @'+str(round(source_sep_lf*100)/100)+'m.<br>The sailline width is '+str(0.5*no_streamers*streamer_sep)+'m.')

    fig.update_layout(title=txt)


    return fig


#Plot the offset distribution on a streamer survey from one shot
def plot_streamer_survey_offset_distribution_from_shot(offsets, no_sources, spi, no_streamers, streamer_sep, streamer_len, layback):
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=offsets,
    xbins=dict( start=0, end=4000, size=round(no_sources*spi*2)  ),
    name='Offset distribution',
    opacity=0.5,
    marker_color='blue'
    ))

    fig.update_layout(
        title_text='Single shot offset distribution on '+str(no_streamers)+' streamers @'+str(streamer_sep)+'m separation and<br>length '+str(streamer_len)+' m. The source layback was '+str(layback)+' m.', # title of plot
        xaxis_title_text='Offset groups of '+str(round(2*spi*no_sources))+' m.', # xaxis label
        yaxis_title_text='TraceCount', # yaxis label
        bargap=0.1, # gap between bars of adjacent location coordinates
        )

    #st.plotly_chart(fig,use_container_width=False)
    return fig

def plot_energy(no_sources, source_sep, spi, no_streamers, streamer_sep, streamer_len, layback, R, S, S_dipole, shooting_area_overlap, critical_reflection_angle):

    [source_x, source_y, streamer_x, streamer_y] = get_source_and_streamer_layout_short(no_sources, source_sep, no_streamers, streamer_sep, streamer_len, layback)
    #get the offsets from the node-survey
    [offsets_streamer, cmp_x_streamer, cmp_y_streamer]=get_offsets_from_one_shot_on_a_spread(source_x, source_y, streamer_x, streamer_y)
    #get the offsets from the streamer survey
    [offsets_node, cmp_x_node, cmp_y_node, node_dx, node_dy, node_area_x, node_area_y, survey_area_x, survey_area_y] = get_offset_from_one_shot_on_nodes(R, S, shooting_area_overlap)

    offsets_streamer = np.sort(offsets_streamer)
    offsets_node = np.sort(offsets_node)

    N=100
    wb = np.linspace(50, streamer_len/2, N) #min,mx and #

    #Computations for the streamer - shooting
    critical_offset = np.zeros((N))
    percetage_useful_energy = np.zeros((N))
    count=0
    for i in range(0,N):
        critical_offset[i] = wb[i]*math.tan(critical_reflection_angle)
        while(offsets_streamer[count]< critical_offset[i] ):
            count = count+1
        percetage_useful_energy[i] = 100*count/len(offsets_streamer)

    #Now a similar computation for the nodes (that sits on the WB) -->offset = twice
    critical_offset_nodes = np.zeros((N))
    percetage_useful_energy_nodes = np.zeros((N))
    count=0
    for i in range(0,N):
        critical_offset_nodes[i] = wb[i]*math.tan(critical_reflection_angle)
        while(2*offsets_node[count]< critical_offset_nodes[i] ): #GO 2x her since node sits on the WB
            count = count+1
        percetage_useful_energy_nodes[i] = 100*count/len(offsets_node)

    #Make the plot for the streamer survey
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wb, y=percetage_useful_energy, mode='markers', name='Streamer', marker_size=6 ))
    fig.add_trace(go.Scatter(x=wb, y=percetage_useful_energy_nodes, mode='markers', name='nodes', marker_size=6))
    txt = ('Percentage of channels with reflection energy from the WB. The critical reflection angle of '+str(round(critical_reflection_angle*180/math.pi))+' degrees.<br>'+str(no_streamers)+' streamers @'+str(streamer_sep)+'m sep. Streamer length='+str(streamer_len)+' m. Layback='+str(layback)+' m.<br>Nodes with spacing (dx, dy)=('+str(round(node_dx))+', '+str(round(node_dy))+') m. The node area: '+str(round(node_area_x/1000))+' X '+str(round(node_area_y/1000))+' km.')
    fig.update_layout(title=txt,  xaxis_title='Water depth',yaxis_title='Percentage (%)')

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=wb[0:15], y=percetage_useful_energy[0:15], mode='markers', name='Streamer', marker_size=6 ))
    fig1.add_trace(go.Scatter(x=wb[0:15], y=percetage_useful_energy_nodes[0:15], mode='markers', name='nodes', marker_size=6))
    fig1.update_layout(title=txt,  xaxis_title='Water depth',yaxis_title='Percentage (%)')

    return [fig, fig1]

#make a roseplot for all sources into a streamer spread
def make_rose_plot_streamer(streamer_x, streamer_y, source_x, source_y, r_alfa, sector_len, mid_x, mid_y):
    #_x: crossline    #_y: inline

    fig = go.Figure()
    #print("mid_x:", mid_x)
    #print("mid_y:", mid_y)
    #print("source_x:", source_x)
    #print("source_y:", source_y)
    #print("streamer_x:", streamer_x)
    #print("streamer_y:", streamer_y)

    layback =  streamer_y[0]- source_y[0]
    #make the polar meshgrid to plot on
    theta = np.linspace(0,360, round(360/r_alfa), endpoint=False)  #start, stop, numb: 6 deg sectors-->60 numb
    #print(theta)
    max_offset = 1.01*((streamer_y[-1] + source_y[0]))
    r=[]  #make the r-range in steps of sector_len
    for i in range(0, 1+math.floor(max_offset/sector_len)):
        r.append(i*sector_len)


    C = np.zeros((len(r), len(theta))) #the count in each r-theta bin
    count=0
    for i in range(0,len(source_x)):
        for j in range(0,len(streamer_y)):
            tmp0 = -(source_y[i]-streamer_y[j])
            tmp2 = tmp0**2
            for k in range(0, len(streamer_x)):
                #find the index with a binary earch to update the C(olor) array
                tmp1=(source_x[i]-streamer_x[k])
                r_index = binarySearch(r,      math.sqrt(tmp1**2 + tmp2 ))
                t_index = binarySearch(theta,  math.atan2(tmp1, tmp0)*180/math.pi + 180)

                C[r_index-1, t_index-1] = C[r_index-1, t_index-1] +1  #NP - subtract 1 due to indexing
                count = count + 1

    #in plotly we need R, Theta and C to be of the same length
    T1 = np.zeros((len(r), len(theta)))
    for i in range(0,len(r)):
        T1[i,:]  = theta
    C=C.flatten()
    T=T1.flatten()+180
    R=sector_len*np.ones((len(r)*len(theta)))

    df = pd.DataFrame({'R':R, 'T':T, 'C':C}) #make a df
    #fig = px.bar_polar(df, r="R", theta="T", color="C",  color_continuous_scale=px.colors.sequential.Jet, title='Roseplot: '+str(len(source_x))+' sources@'+str((source_x[min(1,len(source_x)-1)]-source_x[0]))+'m and '+str(len(streamer_x))+' streamers@'+str(round(streamer_y[-1]-layback))+'m.<br>The source layback was '+str(round(layback))+'m.')
    fig = px.bar_polar(df, r="R", theta="T", color="C", color_continuous_scale=["white","red","orange","yellow","green","cyan","blue","indigo","violet"], title='Roseplot: '+str(len(source_x))+' sources@'+str((source_x[min(1,len(source_x)-1)]-source_x[0]))+'m and '+str(len(streamer_x))+' streamers@'+str(round(streamer_y[-1]-layback))+'m.<br>The source layback was '+str(round(layback))+'m.')
    fig.update_polars(radialaxis_showline=False)
    #fig.update_polars(radialaxis_visible=False)
    #fig.update_polars(radialaxis_title_font_size=40)
    #fig.update_xaxes(title_text="1")
    #fig.update_layout(barmode='group', bargap=0.0, bargroupgap=0.0, radialaxis_visible=False)
    #fig.update_layout(angularaxis)

    return fig

#make a rose plot for a node survey - all shots
def make_rose_plot_nodes(Receiver, S, no_sources, r_alfa, sector_len,  node_dx, node_dy, spi):
    fig = go.Figure()

    #-----------first we plot one shot ---> into all receivers------------------
    #find a source-point in the midle of the spread - and compute all R and theta
    s_x = Receiver[binarySearch(Receiver[:,0], max(Receiver[:,0])/2), 0]
    s_y = Receiver[binarySearch(Receiver[:,1], max(Receiver[:,1])/2), 1]

    source_offset = abs(round(S[round(len(S)/2),1] - S[round(len(S)/2)+1,1]))

    r = np.zeros((len(Receiver)*no_sources))
    t = np.zeros((len(Receiver)*no_sources))
    count=0
    for j in range(0,no_sources):

        for i in range(0, len(Receiver)): #compute all r and theta
            r[count] = math.sqrt( (s_x-Receiver[i][0])**2 + (s_y+j*source_offset-Receiver[i][1])**2  )
            t[count] = math.atan2(s_y-Receiver[i][1], s_x-Receiver[i][0]) + math.pi
            count=count+1

    #Make the R and Theta grid
    max_offset = max(r)

    T = np.linspace(0,2*np.pi, round(360/r_alfa))  #start, stop, numb: 6 deg sectors-->60 numb
    R=[]  #make the r-range in steps of sector_len
    for i in range(0, 1+math.floor(max_offset/sector_len)):
        R.append(i*sector_len)
    C = np.zeros((len(R), len(T))) #the count in each r-theta bin

    for i in range(0, len(Receiver)*no_sources):
        r_index = binarySearch(R, r[i])
        t_index = binarySearch(T, t[i])
        C[r_index, t_index] = C[r_index, t_index] +no_sources

    #in plotly we need R, Theta and C to be of the same size (length)
    T1 = np.zeros((len(R), len(T)))
    for i in range(0,len(R)):
        T1[i,:]  = T

    #then we turn these 2D array into 1D
    C=C.flatten()
    T=T1.flatten()*180/math.pi

    #Remove all entries that are zero in C
    #T=T[C !=0]
    #C=C[C != 0]
    R=sector_len*np.ones((len(T))) #R is just constant (sector_len)

    df = pd.DataFrame({'R':R, 'T':T, 'C':C}) #make a df
    fig = px.bar_polar(df, r="R", theta="T", color="C",   color_continuous_scale=["white","red","orange","yellow","green","cyan","blue","indigo","violet"],title='Roseplot from '+str(no_sources)+' sources@'+str(source_offset)+
      ' shooting in the middle of a receiver spread.<br>'+str(len(Receiver))+
      ' nodes with (Dx, Dy)=('+str(node_dx)+','+str(node_dy)+') on a '+str(round(Receiver[-1,0]/1000))+'x'+str(round(Receiver[-1,1]/1000))+'km grid.'  )

    #--------then make a roseplot for one single receiver-----------------------
    #want to find a node (roughly in the middle of the spread)

    r_x = Receiver[binarySearch(Receiver[:,0], max(Receiver[:,0])/2), 0]
    r_y = Receiver[binarySearch(Receiver[:,1], max(Receiver[:,1])/2), 1]

    r = np.zeros((len(S)))
    t = np.zeros((len(S)))

    for i in range(0, len(S)): #compute all r and theta
            r[i] = math.sqrt( (r_x-S[i][0])**2 + (r_y-S[i][1])**2  )
            t[i] = math.atan2(r_y-S[i][1], r_x-S[i][0]) + math.pi

    #Make the R and Theta grid
    max_offset = max(r)
    T = np.linspace(0,2*np.pi, round(360/r_alfa))  #start, stop, numb: 6 deg sectors-->60 numb
    R=[]  #make the r-range in steps of sector_len
    for i in range(0, 1+math.floor(max_offset/sector_len)):
        R.append(i*sector_len)
    C = np.zeros((len(R), len(T))) #the count in each r-theta bin

    for i in range(0, len(S)):
        r_index = binarySearch(R, r[i])
        t_index = binarySearch(T, t[i])
        C[r_index, t_index] = C[r_index, t_index] +no_sources

    #in plotly we need R, Theta and C to be of the same size (length)
    T1 = np.zeros((len(R), len(T)))
    for i in range(0,len(R)):
        T1[i,:]  = T

    #then we turn these 2D array into 1D
    C=C.flatten()
    T=T1.flatten()*180/math.pi
    R=sector_len*np.ones((len(T))) #R is just constant (sector_len)

    df = pd.DataFrame({'R':R, 'T':T, 'C':C}) #make a df
    fig1 = px.bar_polar(df, r="R", theta="T", color="C",   color_continuous_scale=["white","red","orange","yellow","green","cyan","blue","indigo","violet"],title='Roseplot from '+str(len(S))+' source-points shooting into a receiver near the center of a node spread.<br>Shot grid (Dx, Dy)=('+str(spi)+','+str(source_offset)+') on a '+str(round(S[-1,0]/1000))+'x'+str(round(S[-1,1]/1000))+'km shooting grid.'  )

    return [fig, fig1]

#call the stuff that makes the acq plots
def acq_plots():
    st.title('Survey design assesment:')
    fig = go.Figure()
    fig1 = go.Figure()
    my_expander1 = st.beta_expander("Sources and streamers:", expanded=True)
    no_source_vessels=1
    source_vessel_offset_crossline =0
    source_vessel_offset_inline=0.001
    no_hf_sources_on_vessel1 = 0
    with my_expander1:
        col1, col2, col3, col4 = st.beta_columns(4)

        with col2:
            no_streamers = st.selectbox('Number of streamers' ,[12,0,6,8,10,12,14,16,18,20])
            streamer_sep = st.selectbox('Streamer sep (m)',[100, 50, 62.5, 66.66, 75, 83.33, 90, 100, 110, 120, 150, 200, 300, 400])
            layback = st.selectbox('Source layback (m)',[150, 100, 75, 50, 0, -3000])
            place_lf_in_center = st.checkbox('Place the LF source(s) in the center with the HF or air-guns on the sides', value=False)
        with col3:
            if place_lf_in_center:
                no_hf_sources= st.number_input('Number of HF/air-gun source groups',0,100,2,2)
                no_lf_sources = st.number_input('Number of LF source groups',1,8,2,1)
            else:
                no_hf_sources= st.number_input('Number of HF/air-gun source groups',1,100,2,1)
                no_lf_sources = st.number_input('Number of LF source groups',0,8,0,2)
            no_hf_sources_on_vessel1 = no_hf_sources
            ss = get_valid_source_sep_alternatives(no_hf_sources, streamer_sep)
            source_hf_sep = st.selectbox('HF/air-gun source group sep (m)', ss)
            ss = get_valid_source_sep_alternatives(no_lf_sources, streamer_sep)
            source_lf_sep = st.selectbox('LF source group sep (m)', ss)
        with col1:
            plot_hf_layout = st.button("Plot inline HF/air-gun layout")
            plot_lf_layout = st.button("Plot inline LF layout")
            plot_tot_layout = st.button("Plot inline HF+LF layout")
            plot_hf_birdseye = st.button("Plot birdseye HF+LF layout and HF offset classes")
            make_roseplot =  st.button("Make roseplots")
        with col4:
            extra_source_vessel_1 = st.checkbox('Tick off here for an extra source vessel', value=False)
            if extra_source_vessel_1:
                no_source_vessels=2
                default_offset = round(no_streamers*streamer_sep)
                source_vessel_offset_crossline = st.number_input('Source vessel crossline offset (m)',-20000,20000,default_offset,25)
                source_vessel_offset_inline = st.number_input('Source vessel inline offset (km)',-15,15,1,1)
                source_vessel_offset_inline = source_vessel_offset_inline + 0.001
                source_vessel_no_sources = st.number_input('Number of sources towed by the source vessel',1,9,no_hf_sources,1)
                default_source_sep = streamer_sep/source_vessel_no_sources
                if source_vessel_no_sources==1:
                    default_source_sep=0
                source_vessel_source_sep = st.number_input('Source vessel source separation',default_source_sep, source_vessel_no_sources*default_source_sep, default_source_sep, default_source_sep )


    my_expander0 = st.beta_expander("Extra parameters (sources and streamers):", expanded=False)
    with my_expander0:
        col1, col2, col3 = st.beta_columns(3)

        with col2:
            offset_class_size = st.selectbox('Size of offset class (m)', [100, 50, 75, 100, 125, 150, 200, 250, 300])
            streamer_len = st.selectbox('Streamer length (m)',[6000, 1000, 2000, 3000, 6000, 8000, 9000])
            spi = st.selectbox('Shot-point-interval (flip-to-flop) (m)', [25, 4.17, 6.25, 8.33, 12.50, 16.66, 18.75, 20.00, 25.00, 30, 37.5])
            forced_dy = st.selectbox('Override the nominal dy (m)',[0.0, 5.0, 6.25, 8.33, 10.0, 12.5, 15.0, 18.75, 20.0, 25.0, 30.0, 37.5, 40.0, 50.0])

        with col3:
            r_alfa = st.number_input('Rose plot radial resolution', 1,15,2,1)
            sector_len = st.number_input('Rose plot sector length', 50,1000,200,50)
            extra_sailline_spacing =st.number_input('extra_sailline_spacing', 0,200,0,1)

        #get the source and streamer layout - with the sources in a meaningful position
        [source_lf_x, source_lf_y, source_hf_x, source_hf_y, streamer_x, streamer_y] = get_source_and_streamer_layout(no_lf_sources, source_lf_sep, no_hf_sources, source_hf_sep, no_streamers, streamer_sep, streamer_len, layback, place_lf_in_center)


        if extra_source_vessel_1: #this allows for example side-by-side or TopSeis mode
            [source_vx, source_vy, tmp1,tmp2 ] = get_source_and_streamer_layout_short(source_vessel_no_sources, source_vessel_source_sep, no_streamers, streamer_sep, streamer_len, layback)
            for i in range(0, source_vessel_no_sources):
                source_vx[i] = source_vx[i] + source_vessel_offset_crossline
                source_vy[i] = source_vy[i]  + 1000*source_vessel_offset_inline

            source_hf_x=np.append(source_hf_x, source_vy)
            source_hf_y=np.append(source_hf_y, source_vx)


    my_expander3 = st.beta_expander("Node parameters:", expanded=False)
    with my_expander3:
        col1, col2, col3 = st.beta_columns(3)

        with col1:
            plot_node_layout = st.button("Plot the full node survey layout")
            plot_offset_rec = st.button("Plot node offset receiver distribution")
            compare_offset_node_str = st.button("Compare offset distr: streamers vs nodes")
            make_rose_p_nodes = st.button("Make roseplot nodes")
            plot_offset_shot_node = st.button("Plot offset shot (node)")
            plot_offset_shot_streamer = st.button("Plot offset shot (streamer)")
            plot_percentage_useful_energy = st.button("Reflections vs diffractions")

        with col2:
            survey_area_x = st.number_input('Survey area x (km)',2,100,10,1)
            survey_area_y = st.number_input('Survey area y (km)',2,100,12,1)
            node_dx = st.selectbox('Set nodes DX (m)' ,[100,25,50,100,120,200,400,500,800,1000,1500,2000,3000,4000])
            node_dy = st.selectbox('Set nodes DY (m)' ,[400,25,50,100,120,200,400,500,800,1000,1500,2000,3000,4000])

        with col3:
            shooting_area_overlap = st.number_input('Shooting area outside nodes (km)',0,10,1,1)
            linechange_time       = st.number_input('Linechange time (min)',30,240,180,15)
            vessel_speed = st.number_input('Vessel speed (Kn)',2.5,7.0,4.5,0.1)
            vessel_speed = vessel_speed*0.5144444 #make it m/s
            critical_reflection_angle = st.number_input('Critical ref angle (hard wb --> soft wb)',10,60,40,1)
            critical_reflection_angle = math.pi*critical_reflection_angle/180
            desired_dy = st.selectbox('Desired node dy bin-size (m)',[25.00, 6.25, 7.50, 8.33, 10.00, 12.50, 15.00, 18.75, 25.00, 30.00, 37.50, 50.00])


        #just some stuff me might need later on

        sail_line_sep = no_streamers*streamer_sep/2
        survey_area=survey_area_x*survey_area_y
        node_count=  (1+math.floor(1000*survey_area_x/node_dx))*(1+math.floor(1000*survey_area_y/node_dy))
        bin_size_x= round(100*spi*no_hf_sources/2)/100
        #bin_size_y= desired_dy
        no_saillines = math.ceil((2000*shooting_area_overlap+1000*survey_area_x) / (sail_line_sep))
        survey_time = (3*no_saillines*linechange_time/60) + (3*no_saillines*1000*survey_area_y+2000*shooting_area_overlap)/vessel_speed/3600
        survey_time_dipole = (no_saillines*linechange_time/60) + (no_saillines*1000*survey_area_y+2000*shooting_area_overlap)/vessel_speed/3600

        if (plot_node_layout):
            [R, S] = get_source_and_rec_locations_nodes(survey_area_x, survey_area_y, node_dx, node_dy, source_hf_sep, spi,  no_hf_sources, shooting_area_overlap, False, no_saillines)
            [R, S_dipole] = get_source_and_rec_locations_nodes(survey_area_x, survey_area_y, node_dx, node_dy, source_hf_sep, spi, no_hf_sources, shooting_area_overlap, True, no_saillines)
            [fig, fig1] = plot_node_survey_layout(R, S, S_dipole)
        if (plot_offset_rec):
            [R, S] = get_source_and_rec_locations_nodes(survey_area_x, survey_area_y, node_dx, node_dy, source_hf_sep, spi,no_hf_sources, shooting_area_overlap, False, no_saillines)
            [R, S_dipole] = get_source_and_rec_locations_nodes(survey_area_x, survey_area_y, node_dx, node_dy, source_hf_sep, spi,no_hf_sources, shooting_area_overlap, True, no_saillines)
            fig = plot_node_survey_offset_distribution_on_one_receiver(R, S, S_dipole, spi, no_hf_sources)

        if (compare_offset_node_str):
            [R, S] = get_source_and_rec_locations_nodes(survey_area_x, survey_area_y, node_dx, node_dy, source_hf_sep, spi,no_hf_sources, shooting_area_overlap, False, no_saillines)
            [R, S_dipole] = get_source_and_rec_locations_nodes(survey_area_x, survey_area_y, node_dx, node_dy, source_hf_sep, spi,no_hf_sources, shooting_area_overlap, True, no_saillines)
            fig = plot_compare_streamer_and_node_offset(no_hf_sources, source_hf_sep, spi, no_streamers, streamer_sep, streamer_len, layback, R, S, S_dipole, shooting_area_overlap )
        if make_rose_p_nodes:
            [R, S] = get_source_and_rec_locations_nodes(survey_area_x, survey_area_y, node_dx, node_dy, source_hf_sep, spi,no_hf_sources, shooting_area_overlap, False, no_saillines)
            [fig,fig1]= make_rose_plot_nodes(R, S, no_hf_sources, r_alfa, sector_len, node_dx, node_dy, spi)

        if (plot_offset_shot_node):
            [R, S] = get_source_and_rec_locations_nodes(survey_area_x, survey_area_y, node_dx, node_dy, source_hf_sep, spi,no_hf_sources, shooting_area_overlap, False, no_saillines)
            [R, S_dipole] = get_source_and_rec_locations_nodes(survey_area_x, survey_area_y, node_dx, node_dy, source_hf_sep, spi,no_hf_sources, shooting_area_overlap, True, no_saillines)
            fig = plot_node_survey_offset_distribution_from_shot(R, S, S_dipole, spi, no_hf_sources, shooting_area_overlap)

        if (plot_offset_shot_streamer):
            [source_x, source_y, streamer_x, streamer_y] = get_source_and_streamer_layout_short(no_hf_sources, source_hf_sep, no_streamers, streamer_sep, streamer_len, layback)
            [offsets, cmp_x, cmp_y]=get_offsets_from_one_shot_on_a_spread(source_x, source_y, streamer_x, streamer_y)
            fig = plot_streamer_survey_offset_distribution_from_shot(offsets, no_hf_sources, spi, no_streamers, streamer_sep, streamer_len, layback)


        if plot_percentage_useful_energy:
            [R, S] = get_source_and_rec_locations_nodes(survey_area_x, survey_area_y, node_dx, node_dy, source_hf_sep, spi,no_hf_sources, shooting_area_overlap, False, no_saillines)
            [R, S_dipole] = get_source_and_rec_locations_nodes(survey_area_x, survey_area_y, node_dx, node_dy, source_hf_sep, spi,no_hf_sources, shooting_area_overlap, True, no_saillines)
            [fig, fig1] = plot_energy(no_hf_sources, source_hf_sep, spi, no_streamers, streamer_sep, streamer_len, layback, R, S, S_dipole, shooting_area_overlap, critical_reflection_angle )

    my_expander2 = st.beta_expander("Plots:", expanded=True)
    with my_expander2:

        if (plot_hf_layout):
            [fig, fig1] = plot_layout_(streamer_sep, source_hf_y, source_hf_x, streamer_y, streamer_x,  "HF source layout",no_hf_sources_on_vessel1, no_source_vessels, 0, -0.05, 1.8, extra_sailline_spacing, forced_dy)

        if (plot_lf_layout):
            [fig, fig1] = plot_layout_(streamer_sep, source_lf_y, source_lf_x, streamer_y, streamer_x,  "LF source layout",no_hf_sources_on_vessel1, no_source_vessels)

        if(plot_tot_layout):
            fig = plot_combined_layout_(streamer_sep, source_lf_y, source_lf_x, source_hf_y, source_hf_x, streamer_y, streamer_x, no_source_vessels, no_hf_sources_on_vessel1)

        if(plot_hf_birdseye):
            #combine all source pos ito a single array
            source_y=[]
            source_y.extend(source_hf_y)
            source_y.extend(source_lf_y)

            source_x=[]
            source_x.extend(source_hf_x)
            source_x.extend(source_lf_x)


            hf_source=[]
            for i in range(0, len(source_hf_y)):
                hf_source.append(True)
            for i in range(0, len(source_lf_y)):
                hf_source.append(False)

            bin_hf = "{:3.2f}".format(0.5*streamer_sep/max(no_hf_sources, no_lf_sources))

            fig =  plot_layout_birdseye(streamer_sep, source_y, source_x, streamer_y, streamer_x, hf_source, no_source_vessels, source_vessel_offset_crossline, source_vessel_offset_inline)
            if(no_hf_sources>0 and no_lf_sources>0):
                txt = ('Birdseye view (front end): '+str(no_streamers)+' str @'+str(streamer_sep)+'m. <br>'+str(round(no_hf_sources))+' <b>HF</b> src @'+str(source_hf_sep)+'m and '+str(no_lf_sources)+' LF src @'+str(source_lf_sep)+'m. Effective CMP-width is '+str(0.5*no_streamers*streamer_sep)+' m.<br>The X-line bin sizes (HF, LF) are ('+str(bin_hf)+', '+str(0.5*streamer_sep/no_lf_sources)+')m.')
            elif (no_hf_sources==0 or no_lf_sources>0):
                txt = ('Birdseye view (front end): '+str(no_streamers)+' str @'+str(streamer_sep)+'m. <br>'+str(round(no_lf_sources))+' src @'+str(source_lf_sep)+'m. Effective CMP-width is '+str(0.5*no_streamers*streamer_sep)+' m.<br>The X-line bin size is '+str(0.5*streamer_sep/no_lf_sources)+'m.')
            elif (no_hf_sources>0 or no_lf_sources==0):
                txt = ('Birdseye view (front end): '+str(no_streamers)+' str @'+str(streamer_sep)+'m. <br>'+str(round(no_hf_sources))+' src @'+str(source_hf_sep)+'m. Effective CMP-width is '+str(0.5*no_streamers*streamer_sep)+' m.<br>The X-line bin sizes is '+str(bin_hf)+'m.')

            else:
                txt = ('Birdseye view (front end): '+str(no_streamers)+' str @'+str(streamer_sep)+'m. <br>'+str(round(no_hf_sources/no_source_vessels))+' <b>HF</b> src @'+str(round(source_hf_sep/no_source_vessels))+'m and '+str(no_lf_sources)+' LF src @'+str(round(source_lf_sep/no_source_vessels))+'m. Effective CMP-width is '+str(0.5*no_streamers*streamer_sep)+' m.')
            fig.update_layout(title=txt, xaxis_title='Meter',yaxis_title='Meter', showlegend=False)
            if no_hf_sources>0 and no_streamers>0:
                fig1 = plot_offset_classes(streamer_sep, source_hf_y, source_hf_x, streamer_y, streamer_x, offset_class_size, no_hf_sources_on_vessel1)
                fig1.update_xaxes(range=[streamer_y[0]-50, streamer_y[-1]+50])

        if make_roseplot:
            mid_x = streamer_sep*(no_streamers-1)/2 #the center of the spread - right behind the boat
            mid_y = 0
            if len(source_hf_y) >0:
                fig = make_rose_plot_streamer(streamer_y, streamer_x, source_hf_y, source_hf_x, r_alfa, sector_len, mid_x, mid_y)
            if len(source_lf_y) >0:
                fig1 = make_rose_plot_streamer(streamer_y, streamer_x, source_lf_y, source_lf_x, r_alfa, sector_len, mid_x, mid_y)

        if (len(fig.data)>0):
            st.plotly_chart(fig,use_container_width=False)
        if (len(fig1.data)>0): #then we also plot the second plot
            st.plotly_chart(fig1,use_container_width=False)
    return 0
