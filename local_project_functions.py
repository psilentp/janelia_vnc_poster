import numpy as np

def get_transitions(state_mtrx_dict,muscle_key,pre_trig_idx = 10,post_trig_idx = 100):
    off_on_list = []
    on_off_list = []
    for flynum,tdict in state_mtrx_dict.items():
        for row in tdict[muscle_key]:
            off_on_idx = np.argwhere(np.diff(row[0]) == 1)
            on_off_idx = np.argwhere(np.diff(row[0]) == -1)
            
            for idx in off_on_idx:
                if (idx+post_trig_idx < len(row[0])) & (idx-pre_trig_idx > 0):
                    off_on_list.append((flynum,row[1][idx-pre_trig_idx:idx+post_trig_idx]))
            for idx in on_off_idx:
                if (idx+post_trig_idx < len(row[0])) & (idx-pre_trig_idx > 0):
                    on_off_list.append((flynum,row[1][idx-pre_trig_idx:idx+post_trig_idx]))
    return {'off_on':off_on_list,'on_off':on_off_list}            

def plot_trig_panel(ax_group,
                    trig_key,
                    direction = 'off_on',
                    ts = 0.02,
                    pretrig = 10,
                    posttrig = 50,
                    state_mtrx_dict = None,
                    sorted_keys = None,
                    flydict = None):
    time = np.arange(pretrig+posttrig)*ts
    idx_list = get_transitions(state_mtrx_dict,trig_key,pretrig,posttrig)
    signal_mtrxs = {}
    for key in sorted_keys:
        signal_mtrxs[key] = np.vstack([flydict[fnum].spikestates[key][idx] for fnum,idx in idx_list[direction]])
    signal_mtrxs['left','amp'] =  np.hstack([np.array(flydict[fnum].left_amp)[idx] for fnum,idx in idx_list[direction]]).T
    signal_mtrxs['right','amp'] =  np.hstack([np.array(flydict[fnum].right_amp)[idx] for fnum,idx in idx_list[direction]]).T
    filter_cond = np.sum(signal_mtrxs[trig_key][:,:11],axis = 1) == {'off_on':0,'on_off':1*11}[direction]
    rast_mtrx = signal_mtrxs[trig_key][filter_cond,:]
    rast_mtrx = rast_mtrx[np.random.randint(0,rast_mtrx.shape[0],size = 100),:]
    ax_group['raster'].imshow(rast_mtrx,
               aspect = 'auto',interpolation = 'nearest',extent = [0,time[-1],0,100])
    ax_group['raster'].set_ybound(0,100)
    ax_group['kine'].plot(time,np.rad2deg(np.nanmean(signal_mtrxs['left','amp'][filter_cond,:],axis = 0)))
    ax_group['kine'].plot(time,np.rad2deg(np.nanmean(signal_mtrxs['right','amp'][filter_cond,:],axis = 0)))
    for key,ax in ax_group['left'].items():
        ax.plot(time,np.nanmean(signal_mtrxs['left',key][filter_cond,:],axis = 0),color = 'b')
    for key,ax in ax_group['right'].items():
        ax.plot(time,np.nanmean(signal_mtrxs['right',key][filter_cond,:],axis = 0),color = 'g')


def make_state_matrix(flylist,
                     sorted_keys,
                     block_key = 'cl_blocks, g_x=-1, g_y=0 b_x=0, b_y=0'):
    state_mtrxs = []
    left = []
    right = []
    lmr = []
    stim_key = ('common','idx',block_key)
    for fly in flylist:
        state_mtrx = np.vstack([fly.spikestates[key] for key in sorted_keys])
        #key = ('common', 'idx', 'cl_blocks, g_x=-1, g_y=0 b_x=-8, b_y=0')
        #key = ('common', 'idx', 'cl_blocks, g_x=-1, g_y=0 b_x=8, b_y=0')
        idx_list = fly.block_data[stim_key]
        state_mtrxs.extend([np.array(state_mtrx[:,idx[100:]]) for idx in idx_list])
        left.extend([np.array(fly.left_amp)[idx[100:]] for idx in idx_list])
        right.extend([np.array(fly.right_amp)[idx[100:]] for idx in idx_list])
        lmr.extend([np.array(fly.left_amp)[idx[100:]]-np.array(fly.right_amp)[idx[100:]] 
                    for idx in idx_list])
    state_mtrx = np.hstack(state_mtrxs)
    state_mtrx = state_mtrx.astype(int)
    return state_mtrx,np.vstack(left),np.vstack(right)

def get_transiton_prob(state_mtrx):
    tprob = {}
    state_list = [tuple(row) for row in state_mtrx.T]
    state_set = set(state_list)
    state_counts = {}
    for state in state_set:
        state_counts[state] = np.sum(np.sum(state==state_mtrx.T,axis = 1)==8)
    tprob = {}
    for col1,col2 in zip(state_mtrx.T[:-1],state_mtrx.T[1:]):
        if (tuple(col1),tuple(col2)) in tprob.keys():
            tprob[tuple(col1),tuple(col2)] += 1
        else:
            tprob[tuple(col1),tuple(col2)] = 1
    return tprob,state_counts

def make_transition_matrix(tprob,
                           state_counts,
                           min_tran_num = 1,
                           min_state_num = 10):
    filtered = {}
    for key,tnum in tprob.items():
        if (tnum > min_tran_num) & \
              (state_counts[key[0]] > min_state_num) & \
              (state_counts[key[1]]> min_state_num):
            filtered[key] = tnum

    inkeys = [x[0] for x in filtered.keys()]
    outkeys = [x[1] for x in filtered.keys()]

    filterd_set = list(set(inkeys + outkeys))
    transition_mtrx = np.zeros((len(filterd_set),len(filterd_set)))

    for i,state1 in enumerate(filterd_set):
        for j,state2 in enumerate(filterd_set):
            try:
                transition_mtrx[i,j] = filtered[state1,state2]
            except KeyError:
                pass

    transition_mtrx = transition_mtrx/np.sum(transition_mtrx,axis = 1)[:,None]
    transition_mtrx[np.isnan(transition_mtrx)] = 0
    sidx = np.argsort(np.diag(transition_mtrx))[::-1]
    transition_mtrx = transition_mtrx[sidx].T[sidx]
    state_table = np.array(filterd_set)[sidx,:]
    return transition_mtrx,state_table

def next_state(current_state,state_table,tmtrx):
    """simulate a markov step using transition matrx"""
    from numpy import random
    state_idx = np.squeeze(np.argwhere(np.all(state_table == current_state,axis = 1)))
    #print state_idx
    prob_vector = tmtrx[:,state_idx]
    #print prob_vector
    idx = random.choice(np.arange(len(state_table)),p = prob_vector)
    return state_table[idx]

def key_to_key(inkey):
    """map the input key from the cov_mtrx_triang_layout.svg into a set of keys 
    (a tuple of tuples) that can be used to construct signals from the fly data"""
    try:
        k1 = {'R':'right','L':'left'}[inkey[0]]
    except KeyError:
        if inkey[0] == 'w':
            return ('common','wb_freq')
        else:
            return
    k2 = inkey[1:]
    return (k1,k2)
            
def make_scatter_plots(fly,ax_group):
    """create summary plot for each fly in flylist, uses the template 
    file cov_matrix_triang_layout.svg, the hdf5 data of each fly needs to 
    be loaded"""
    import figurefirst as fifi
    if 'data_mask' in fly.h5files.keys():
        #fifi.mpl_functions.kill_all_spines(layout)
        dmask = np.array(fly.data_mask)

        # keymap = {}
        #l = [keymap.update({key:(key_to_key(key[0]),key_to_key(key[1]))}) for key in layout.axes.keys()]
        #keymap.pop('flynum')
        for key1,g in ax_group.items():
            for key2,ax in g.items():
                try:
                    dkeys = (key_to_key(key1),key_to_key(key2))
                    dta1 = fly.non_neg_signals[dkeys[0]][dmask][::50]
                    dta2 = fly.non_neg_signals[dkeys[1]][dmask][::50]
                    ax.scatter(dta1,dta2,
                               marker = '.',
                               s = 0.5,
                               color = 'k',
                               alpha = 0.5,
                               edgecolors = 'none',
                               rasterized = True)
                    ax.set_xbound(0,1)
                    ax.set_ybound(0,1)
                    fifi.mpl_functions.kill_spines(ax)
                except KeyError:
                    if key == 'flynum':
                        ax.text(0,0,'Fly%s'%fly.flynum,size = 20)
                    else:
                        pass
            #layout.save('scatter_matrix_%s.svg'%fly.flynum)
            #plt.close('all')
        #display(SVG('scatter_matrix_%s.svg'%fly.flynum))
        
def get_kde_estimates(flies,xkey,ykey,
                      condition_keys = [],
                      block_keys = [],
                      nbins = 50,bandwidth = 0.05):
    from sklearn.neighbors.kde import KernelDensity
    pdfs = []
    for fly in flies:
        try:
            dmask = fly.data_mask
            c_masks = []
            b_masks = []

            for ckey in condition_keys:
                c_masks.append(np.array(fly.experimental_condition) == ckey)
            for bkey in block_keys:
                b_masks.append(np.array(fly.experimental_block) == bkey)
            if len(c_masks)>0:
                if len(b_masks)>0:
                    dmask = dmask & np.any(b_masks,axis = 0) & np.any(c_masks,axis = 0)
                else:
                    dmask = dmask & np.any(c_masks,axis = 0)
            else:
                if len(b_masks)>0:
                    dmask = dmask & np.any(b_masks,axis = 0)

            x = fly.non_neg_signals[xkey][dmask]
            y = fly.non_neg_signals[ykey][dmask]
            X = np.array([x,y])
            xedges = np.linspace(0,1,nbins)
            yedges = np.linspace(0,1,nbins)
            gridx,gridy = np.meshgrid(xedges,yedges)
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X.T)
            smps = kde.score_samples(np.vstack((gridx.ravel(),gridy.ravel())).T)
            pdfs.append((np.exp(smps.reshape(np.shape(gridx)))))
        except ValueError:
            pass
    return pdfs

def plot_sector(flylist,
                xkey,ykey,
                sector,ax,
                pathspecs,
                mode = 'hist',
                contours = False):
    sector_map = {'rb-tu':'cl_blocks, g_x=-1, g_y=4 b_x=8, b_y=0',
                  'nb-tu':'cl_blocks, g_x=-1, g_y=4 b_x=0, b_y=0',
                  'lb-tu':'cl_blocks, g_x=-1, g_y=4 b_x=-8, b_y=0',
                  
                  'rb-nu':'cl_blocks, g_x=-1, g_y=0 b_x=8, b_y=0',
                  'nb-nu':'cl_blocks, g_x=-1, g_y=0 b_x=0, b_y=0',
                  'lb-nu':'cl_blocks, g_x=-1, g_y=0 b_x=-8, b_y=0',
                  
                  'rb-td':'cl_blocks, g_x=-1, g_y=-4 b_x=8, b_y=0',
                  'nb-td':'cl_blocks, g_x=-1, g_y=-4 b_x=0, b_y=0',
                  'lb-td':'cl_blocks, g_x=-1, g_y=-4 b_x=-8, b_y=0'}
    
    block_key = sector_map[sector]
    if mode == 'hist':
        group_hist = get_group_hist(flylist,
                                 xkey,
                                 ykey,
                                 block_keys = [block_key],
                                 condition_keys = ['condition=test'])
        sector_mask = np.array([group_hist>0.5]*4)
        ps = pathspecs[sector]
        fc = ps.mplkwargs()['facecolor']
        sector_img = (sector_mask*fc[:,None,None]).T
        ax.imshow(sector_img,interpolation = 'None',extent = (0,1,0,1))
    if mode == 'kde':
        group_hists = get_kde_estimates(flylist,
                                 xkey,
                                 ykey,
                                 block_keys = [block_key],
                                 condition_keys = ['condition=test'],
                                 bandwidth = 0.1,
                                 nbins = 100)
        group_hists = np.product(group_hists,axis = 0)
        if contours:
            #sector_mask = np.array(group_hists>0.001*4)
            #pad the mask to deal with corners and edges
            sector_mask = group_hists
            padr = np.zeros(np.shape(sector_mask)[1])
            sector_mask = np.vstack((padr,sector_mask,padr))
            padc = np.zeros((np.shape(sector_mask)[0],1)) 
            sector_mask = np.hstack((padc,sector_mask,padc))
            #get the contour
            from skimage import measure
            contours = measure.find_contours(sector_mask,30.5)
            if len(contours) > 1:
                print('more than one contour found for:')
                print('%s %s - %s %s'%(xkey + ykey))
                contour = contours[np.argmax([len(c) for c in contours])].T
            else:
                contour = contours[0].T
            #put back in data coords
            cy = (contour[0]-1)/(np.shape(sector_mask)[0]-1)
            cx = (contour[1]-1)/(np.shape(sector_mask)[1]-1)
            if (cy[0] != cy[-1]) | (cx[0] != cx[-1]):
                #close the contours
                cy = np.hstack([cy,cy[0]])
                cx = np.hstack([cx,cx[0]])
            ps = pathspecs[sector]
            #fc = ps.mplkwargs()['facecolor']
            #ec = ps.mplkwargs()['edgecolor']
            kwargs = ps.mplkwargs()
            ax.fill(cx,cy,clip_on = False,**kwargs)
        else:
            sector_mask = np.array([group_hists>0.001]*4)
            ps = pathspecs[sector]
            fc = ps.mplkwargs()['facecolor']
            sector_img = np.transpose(sector_mask*fc[:,None,None],(1,2,0))[::-1,:,:]
            ax.imshow(sector_img,interpolation = 'None',extent = (0,1,0,1))

def get_kde_estimates_kine(flies,rng = (25,95),
                      condition_keys = [],
                      block_keys = [],
                      nbins = 50,bandwidth = 10):
    from sklearn.neighbors.kde import KernelDensity
    from thllib import util
    pdfs = []
    for fly in flies:
        #try:
        dmask = fly.data_mask
        c_masks = []
        b_masks = []

        for ckey in condition_keys:
            c_masks.append(np.array(fly.experimental_condition) == ckey)
        for bkey in block_keys:
            b_masks.append(np.array(fly.experimental_block) == bkey)
        if len(c_masks)>0:
            if len(b_masks)>0:
                dmask = dmask & np.any(b_masks,axis = 0) & np.any(c_masks,axis = 0)
            else:
                dmask = dmask & np.any(c_masks,axis = 0)
        else:
            if len(b_masks)>0:
                dmask = dmask & np.any(b_masks,axis = 0)

        x = util.fill_nan(np.rad2deg(np.array(fly.left_amp)[dmask]))
        y = util.fill_nan(np.rad2deg(np.array(fly.right_amp)[dmask]))
        X = np.array([x,y])
        X[np.isnan(X)] = 0
        X = np.squeeze(X)
        xedges = np.linspace(rng[0],rng[1],nbins)
        yedges = np.linspace(rng[0],rng[1],nbins)
        gridx,gridy = np.meshgrid(xedges,yedges)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X.T)
        smps = kde.score_samples(np.vstack((gridx.ravel(),gridy.ravel())).T)
        pdfs.append((np.exp(smps.reshape(np.shape(gridx)))))
        #except ValueError:
        #    pass
    return pdfs

def plot_sector_kine(flylist,
                sector,ax,
                pathspecs,
                mode = 'hist',
                contours = False,
                rng = (25,95)):
    sector_map = {'rb-tu':'cl_blocks, g_x=-1, g_y=4 b_x=8, b_y=0',
                  'nb-tu':'cl_blocks, g_x=-1, g_y=4 b_x=0, b_y=0',
                  'lb-tu':'cl_blocks, g_x=-1, g_y=4 b_x=-8, b_y=0',

                  'rb-nu':'cl_blocks, g_x=-1, g_y=0 b_x=8, b_y=0',
                  'nb-nu':'cl_blocks, g_x=-1, g_y=0 b_x=0, b_y=0',
                  'lb-nu':'cl_blocks, g_x=-1, g_y=0 b_x=-8, b_y=0',

                  'rb-td':'cl_blocks, g_x=-1, g_y=-4 b_x=8, b_y=0',
                  'nb-td':'cl_blocks, g_x=-1, g_y=-4 b_x=0, b_y=0',
                  'lb-td':'cl_blocks, g_x=-1, g_y=-4 b_x=-8, b_y=0'}

    block_key = sector_map[sector]

    if mode == 'hist':
        group_hist = get_group_hist(flylist,
                                 xkey,
                                 ykey,
                                 block_keys = [block_key],
                                 condition_keys = ['condition=test'])
        sector_mask = np.array([group_hist>0.5]*4)
        ps = pathspecs[sector]
        fc = ps.mplkwargs()['facecolor']
        sector_img = (sector_mask*fc[:,None,None]).T
        ax.imshow(sector_img,interpolation = 'None',extent = (0,1,0,1))
    if mode == 'kde':
        group_hists = get_kde_estimates_kine(flylist,rng = rng,
                                             block_keys = [block_key],
                                 condition_keys = ['condition=test'],
                                 bandwidth = 5,
                                 nbins = 100)
        print np.shape(group_hists)
        group_hists = np.product(group_hists,axis = 0)
        if contours:
            #sector_mask = np.array(group_hists>0.001*4)
            #pad the mask to deal with corners and edges
            sector_mask = group_hists
            padr = np.zeros(np.shape(sector_mask)[1])
            sector_mask = np.vstack((padr,sector_mask,padr))
            padc = np.zeros((np.shape(sector_mask)[0],1)) 
            sector_mask = np.hstack((padc,sector_mask,padc))
            #get the contour
            from skimage import measure
            level = np.percentile(group_hists,90)
            contours = measure.find_contours(sector_mask,level)
            if len(contours) > 1:
                print('more than one contour found')
                contour = contours[np.argmax([len(c) for c in contours])].T
            else:
                contour = contours[0].T
            #put back in data coords
            cy = (contour[0]-1)/(np.shape(sector_mask)[0]-1)
            cx = (contour[1]-1)/(np.shape(sector_mask)[1]-1)
            
            cy =cy*(rng[1]-rng[0]) + rng[0]
            cx =cx*(rng[1]-rng[0]) + rng[0]
            
            if (cy[0] != cy[-1]) | (cx[0] != cx[-1]):
                #close the contours
                cy = np.hstack([cy,cy[0]])
                cx = np.hstack([cx,cx[0]])
            ps = pathspecs[sector]
            kwargs = ps.mplkwargs()
            ax.fill(cx,cy,clip_on = False,**kwargs)
        else:
            sector_mask = np.array([group_hists>0.001]*4)
            ps = pathspecs[sector]
            fc = ps.mplkwargs()['facecolor']
            sector_img = np.transpose(sector_mask*fc[:,None,None],(1,2,0))[::-1,:,:]
            ax.imshow(sector_img,interpolation = 'None',extent = (0,1,0,1))
            
def make_single_kernel(times,tauon1,tauoff1):
    kx = np.copy(times)
    kon1 = lambda x:np.exp(((-1*tauon1)/(x)))
    koff1 = lambda x:np.exp((-1*x)/tauoff1)
    k1 = (kon1(kx)*koff1(kx))
    return k1/np.max(k1)