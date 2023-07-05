import numpy as np
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

class plot_utils:
    def __init__(self):
        return
    
#Code courtesy of Brendan Artley
    def hex_to_RGB(self,hex_str):
        """ #FFFFFF -> [255,255,255]"""
        #Pass 16 to the integer function for change of base
        return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]
    def get_color_gradient(self,c1, c2, n):
        """
        Given two hex colors, returns a color gradient
        with n colors.
        """
        assert n > 1
        c1_rgb = np.array(self.hex_to_RGB(c1))/255
        c2_rgb = np.array(self.hex_to_RGB(c2))/255
        mix_pcts = [x/(n-1) for x in range(n)]
        rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
        return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]
#End of code courtesy of Brendan Artley

    def default_dataset_dictionary(self):
        return {'type':'plot','markersize':1,'markercolor':'#070808','linewidth':1,'linecolor':'#070808','dataset_name':'default','bar_width':1,'linestyle':'-','marker':'o','colormap':'magma','alpha':1}

    def default_subplot_dictionary(self):
        return {'x_label':None,'y_label':None,'title':None,'logx':False,'logy':False,'legend':False,'xticklabels':None,'xticklocations':None,'xtickrotation':0,
                'yticklabels':None,'yticklocations':None,'ytickrotation':0,'axiscolor':'black','tickfontcolor':'black','labelfontcolor':'black','labelfontsize':20,
                'tickfontsize':10,'legendfontsize':10,'legendfontcolor':'black','xlims':None,'ylims':None,'patches':[],'noYticks':False,'noXticks':False,'colorbar':False,
                'zlims':[None,None],'colorbarLabel':'','colorbarNorm':'linear','axiswidth':0.8,'tickwidth':0.8,'ticklength':3.5,'legendframe':True,'contourlevels':50,'legendlocation':'best'}

    def default_global_dictionary(self):
        return {'save':False,'save_name':'test.png','size':(8,10),'show':True,'dpi':100,'bbox_inches':None,'tight_layout':True,'subplots_adjust':{'left':None,'right':None,'top':None,'bottom':None,'hspace':None,'wspace':None},'height_ratios':None}

    def clear_plot(self):
        try:
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()
        except:
            pass
        return
    
    #Python recently added swanky new dictionary merger syntax, so it seems like a good time to take another crack at a universal multiplot function. Will grow as I decide I need more control over things.
    #x_sets=list (len N) of lists (len M); one list per subplot. Within each subplot list, gives x coords of M different lines. If len(x_sets[n])!=len(y_sets[n]), the x_sets[n][m=00] will be used for all M lists in y_sets[n]
    #y_sets=list (len N) of lists (len M); one list per subplot. Within each subplot list, gives y coords of M different lines.
    #dataset_dictionaries=list (len N) of lists (len M) of dictionaries specifying dataset-specific parameters. Currently accepts 'type' ('scatter', 'plot', or 'bar'), 'markersize', 'markercolor', 'linewidth','linecolor','dataset_name','bar_width' (for bar plots)
    #subplot_dictionaries=list (len N) of dictionaries specifying subplot-specific parameters. Currently accepts 'x_label', 'y_label', 'title', 'logx', 'logy', 'ticklabels', 'ticklocations', 
    #global_dicitonary=dictionary of global params. Currently accepts 'save','save_name','size' (two-tuple),'show','dpi'
    def multiplot(self,x_sets,y_sets,dataset_dictionaries,subplot_dictionaries,global_dictionary):
        default_dd=self.default_dataset_dictionary()
        default_sd=self.default_subplot_dictionary()
        default_gd=self.default_global_dictionary()
        gd={**default_gd, **global_dictionary}
        N=len(y_sets)
        fig,ax=plt.subplots(nrows=N,ncols=1,figsize=gd['size'],squeeze=False,dpi=gd['dpi'],sharex=gd['sharex'],gridspec_kw={'height_ratios':gd['height_ratios']})
        cbarwidths=[]
        for n in range(N):
            cur_ys=y_sets[n]
            cur_xs=x_sets[n]
            cur_sd= {**default_sd, **subplot_dictionaries[n]}
            M=len(cur_ys)
            if len(cur_xs)!=M:
                cur_xs=list([cur_xs[0] for i in range(N)])
            for m in range(M):
                cur_dd={**default_dd, **dataset_dictionaries[n][m]}
                if cur_dd['type']=='plot':
                    ax[n,0].plot(cur_xs[m],cur_ys[m],color=cur_dd['linecolor'],linewidth=cur_dd['linewidth'],markersize=cur_dd['markersize'],markerfacecolor=cur_dd['markercolor'],linestyle=cur_dd['linestyle'],marker=cur_dd['marker'],label=cur_dd['dataset_name'],markeredgewidth=0,alpha=cur_dd['alpha'])
                elif cur_dd['type']=='scatter':
                    ax[n,0].scatter(cur_xs[m],cur_ys[m],s=cur_dd['markersize'],c=cur_dd['markercolor'],marker=cur_dd['marker'],label=cur_dd['dataset_name'],alpha=cur_dd['alpha'],linewidths=cur_dd['linewidth'])
                elif cur_dd['type']=='bar':
                    ax[n,0].bar(cur_xs[m],cur_ys[m],width=cur_dd['bar_width'],color=cur_dd['linecolor'],label=cur_dd['dataset_name'])
                elif cur_dd['type']=='pcolormesh':
                    pcm=ax[n,0].pcolormesh(cur_ys[m][0],cur_ys[m][1],cur_xs[m],cmap=cur_dd['colormap'], vmin = cur_sd['zlims'][0], vmax = cur_sd['zlims'][1],rasterized=True,norm=cur_sd['colorbarNorm'])
                elif cur_dd['type']=='contourf':
                    pcm=ax[n,0].contourf(cur_ys[m][0],cur_ys[m][1],cur_xs[m],cmap=cur_dd['colormap'], levels=cur_sd['contourlevels'],vmin = cur_sd['zlims'][0], vmax = cur_sd['zlims'][1],norm=cur_sd['colorbarNorm'])
                    for c in pcm.collections:
                        c.set_rasterized(True)
                if cur_sd['colorbar']:
                    divider = make_axes_locatable(ax[n,0])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar=fig.colorbar(pcm, cax=cax, orientation='vertical')
                    cbar.ax.tick_params(which='both',color=cur_sd['axiscolor'],labelsize=cur_sd['tickfontsize'],labelcolor=cur_sd['tickfontcolor'],length=cur_sd['ticklength'],width=cur_sd['tickwidth'])
                    cbar.outline.set_color(cur_sd['axiscolor'])
                    cbar.outline.set_linewidth(cur_sd['axiswidth'])
                    cbar.ax.set_ylabel(cur_sd['colorbarLabel'],color=cur_sd['labelfontcolor'],fontsize=cur_sd['labelfontsize'],fontfamily='sans-serif')                
                    cbar_pos=cax.get_position()
                    cbarwidths.append([n,cbar_pos.width])
            if cur_sd['xlims']!=None:
                ax[n,0].set_xlim(cur_sd['xlims'][0],cur_sd['xlims'][1])
            if cur_sd['ylims']!=None:
                ax[n,0].set_ylim(cur_sd['ylims'][0],cur_sd['ylims'][1])
            for patch in cur_sd['patches']:
                ax[n,0].add_patch(patch)
            if cur_sd['logx']:
                ax[n,0].set_xscale('log')
            if cur_sd['logy']:
                ax[n,0].set_yscale('log')
            if cur_sd['xticklocations']!=None:
                ax[n,0].set_xticks(cur_sd['xticklocations'],cur_sd['xticklabels'])
                ax[n,0].tick_params(axis='x',labelrotation=cur_sd['xtickrotation'])
                ax[n,0].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            if cur_sd['yticklocations']!=None:
                ax[n,0].set_yticks(cur_sd['yticklocations'],cur_sd['yticklabels'])
                ax[n,0].tick_params(axis='y',labelrotation=cur_sd['ytickrotation'])
                ax[n,0].get_yaxis().set_major_formatter(ticker.ScalarFormatter())
            if cur_sd['noYticks']:
                ax[n,0].set_yticks([])
                ax[n,0].tick_params(axis='y',labelleft=False)
            if cur_sd['noXticks']:
                ax[n,0].set_xticks([])
                ax[n,0].tick_params(axis='x',labelbottom=False)
            ax[n,0].tick_params(axis='both',which='both',color=cur_sd['axiscolor'],labelsize=cur_sd['tickfontsize'],labelcolor=cur_sd['tickfontcolor'],length=cur_sd['ticklength'],width=cur_sd['tickwidth'])
            for spine in ['top','bottom','left','right']:
                ax[n,0].spines[spine].set_color(cur_sd['axiscolor'])
                ax[n,0].spines[spine].set_linewidth(cur_sd['axiswidth'])
            if cur_sd['legend']:
                ax[n,0].legend(fontsize=cur_sd['legendfontsize'],labelcolor=cur_sd['legendfontcolor'],frameon=cur_sd['legendframe'],loc=cur_sd['legendlocation'])
            if cur_sd['x_label']!=None:
                ax[n,0].set_xlabel(cur_sd['x_label'],color=cur_sd['labelfontcolor'],fontsize=cur_sd['labelfontsize'],fontfamily='sans-serif')
            if cur_sd['y_label']!=None:
                ax[n,0].set_ylabel(cur_sd['y_label'],color=cur_sd['labelfontcolor'],fontsize=cur_sd['labelfontsize'],fontfamily='sans-serif')
            if cur_sd['title']!=None:
                ax[n,0].set_title(cur_sd['title'],color=cur_sd['labelfontcolor'],fontsize=cur_sd['labelfontsize'],fontfamily='sans-serif')
        if gd['sharex'] and len(cbarwidths)>0:
            ax=self.align_axes(ax,N,cbarwidths)
        if gd['tight_layout']:
            fig.canvas.draw()
            fig.tight_layout()
        plt.subplots_adjust(top=gd['subplots_adjust']['top'],bottom=gd['subplots_adjust']['bottom'],right=gd['subplots_adjust']['right'],left=gd['subplots_adjust']['left'],hspace=gd['subplots_adjust']['hspace'],wspace=gd['subplots_adjust']['wspace'])
        if gd['save']:
            plt.savefig(gd['save_name'], dpi=gd['dpi'],transparent=True)
        if gd['show']:
            plt.show()
        return

    def align_axes(self,ax,N,cbarwidths):
        cbar_inds=[i[0] for i in cbarwidths]
        for n in range(N):
            if n not in cbar_inds:
                divider = make_axes_locatable(ax[n,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cax.remove()
        return ax


class pickle_utils:
    def __init__(self):
        return

    def pu_pickle_evo(self,name,to_export,dir="/content/drive/MyDrive/tmm_output/"):
        f=open(dir+name+".txt",'wb')
        pickle.dump(to_export,f)
        f.close()
        return

    def pu_import_evo(self,file,attrName,dir="/content/drive/MyDrive/tmm_output/"):
        f=open(dir+file+".txt",'rb')
        setattr(self,attrName,pickle.load(f))
        f.close()
        return

class comp_utils:
    def __init__(self):
        return
    #Fast bisection search which finds the indices of the elements in lis which most strictly bound val. If val lies beyond lis's min/max, will return the first two / last two indices, respectively
    #If val is exactly equal to an element of lis, will return that index and the next index.
    def findBoundInds(self,lis,val):
        lbnd=0
        ubnd=len(lis)-1
        while ubnd-lbnd!=1:
            cur_trial=int((ubnd+lbnd)/2)
            cur_val=lis[cur_trial]
            if val>=cur_val:
                lbnd=cur_trial
            else:
                ubnd=cur_trial
        return lbnd,ubnd

    def ripple_add(self,add_this,to_this,this_many_times):
        for i in range(this_many_times):
            to_this.append(add_this)
        return to_this

    def mul(self,v1,v2):
        if hasattr(v1,'__iter__'):
            if hasattr(v2,'__iter__'):
                return np.array([self.mul(v1[i],v2[i]) for i in range(len(v1))])
            else:
                return np.array([self.mul(v1[i],v2) for i in range(len(v1))])
        elif hasattr(v2,'__iter__'):
            return np.array([self.mul(v2[i],v1) for i in range(len(v2))])
        else:
            if v2==0 or v1==0:
                return 0
            else:
                return v1*v2

    def log10(self,val):
        if val==0:
            return -np.inf
        elif val==np.inf:
            return np.inf
        else:
            return np.log10(val)

    def div(self,num,denom):
        if hasattr(num,'__iter__'):
            if hasattr(denom,'__iter__'):
                return np.array([self.div(num[i],denom[i]) for i in range(len(num))])
            else:
                return np.array([self.div(num[i],denom) for i in range(len(num))])
        elif hasattr(denom,'__iter__'):
            return np.array([self.div(num,denom[i]) for i in range(len(denom))])
        else:
            if num==0 or denom==np.inf or denom==-np.inf or denom!=denom:
                return 0
            elif denom==0:
                return np.inf
            else:
                return num/denom
                
    def listdel(self,list,ind):
        list.pop(ind)
        return list
        
    def tupledel(self,tup,ind):
        tup=list(tup)
        tup.pop(ind)
        return tuple(tup)
                
    def arraydel(self,arr,ind):
        deltools={"<class 'numpy.ndarray'>":np.delete,"<class 'list'>":self.listdel,"<class 'tuple'>":self.tupledel}
        return deltools[str(type(arr))](arr,ind)