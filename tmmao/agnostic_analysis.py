import numpy as np
import matplotlib.patches as patches
from misc_utils import plot_utils,pickle_utils,comp_utils
from agnostic_invDes import agnostic_invDes

#Dumping ground for analysis methods because agnostic_director finally topped 1k lines and that's my cutoff.
class agnostic_analysis(pickle_utils,plot_utils,comp_utils):
    def __init__(self):
        return

    #Loads a previous optimization evolution dictionary for further analysis.
    #name=file name, sans director, with extension
    #dir=directory, with final seperation character
    def load_evo(self,name,dir):
        self.evo.clear()
        self.pu_import_evo(name,'evo',dir=dir)
        self.grand_iters=list(self.evo.keys())
        self.grand_iters.sort()
        del self.grand_iters[0]#First grand_iter is -1, the preamble. This is not a real grand_iter, so get rid of it
        self.physics=self.evo[-1]['physics']
        self.sim_params['physics']=self.physics
        return

#Ok, here's how analysis is supposed to work. Since analysis requires the same simulation that optimization does, it makes sense to put them in the same module. That's why it's here.
#If you want to do analysis without doing an optimization, use load_evo() to load in an old evo dictionary. Then call set_physics(), since we clearly need physics functions to do simulating.
#Next, call init_analysis() to load in simulation parameters like the sweep range, substrates, the simulation type, etc. If you just did an optimizaiton, you can skip these steps, or you can call
#only init_analysis if you wish to use a different simulation range, scale, etc. than you did for the optimization.
#Regardless, now you can do analyzing. First, call get_indicatorsEvo() to get data on the structure at a chosen iteration, or get_indicatorsCustom() to get data on a custom structure
#These accept a list of strings, each an "indicator" that the physics package will compute given a set of fields and material parameters. These indicators are any value which indicates how well the device is performing.
#They could be scalars, like the total transmission loss through an acoustic device, or vectors, like the reflectance of an optical device as a function of frequency. Computation of these indicators is done entirely 
#within the physics package, after the simulation has been done here. The physics package must have a function
#   indicators(indicators,simPoints,third_vars,simScale,fields,mat1params,mat2params,param0s,paramNs,customInput,name)={'name of a value to print/plot':([the value, either a scalar or 1D array-like],'the units of the value, to be dispalyed'),...}
#       indicators: array-like of strings. Each string should be recognized by indicators, and correspond to a particular value to output
#       simPoints: array-like. The simulation points, in scaled units
#       third_vars: array-like of dictionaries of third variables, like temperature, pressure, incident angle, incident polarization, etc.
#       simScale: float, the conversion factor from simulation scale to SI units
#       fields: array-like of array-like of vectors. fields[i][j][k] is the kth element of the statevector in layer j (j=0 being the incidence medium) at (thrid_var,simPoint) combination i
#       mat1params: array-like of dictionaries. mat1params[i] is the dictionary of properties of material 1 at (thrid_var,simPoint) combination i
#       mat2params: array-like of dictionaries. As above, but for material 2
#       param0s: array-like of some kind of object. For simType='fixedPoint', should be array-like of scalars, the y-values of the first stationary anchor points, at (thrid_var,simPoint) combination i. For simType=='independent',
#                should be array-like of incidence material dictionaries at (thrid_var,simPoint) combination i
#       paramNs: array-like of some kind of object. As above, but for transmission medium.
#       customInput: array-like of some kind of object. Extra values that were computed at each (thrid_var,simPoint) combination. For example, these could be an incident noise spectrum, to be used to find full-spectrum transmission loss
#       name: string to be appended to the end of each key. To differentiate same indicators from different structures
#If the physics package has no such method, then it can be used for optimization, but not analysis.
#There are no restrictions on how many or how few entries the return dictionary may have. However, the returned dictionary will be merged with the dictionaries returned by all previous calls of either get_indicatorsEvo() or get_indicatorsCustom(),
#folded into the master self.indicator_dict dictionary, so new entries with the same keys as older entries will overwrite the old entries. Once you've computed all the indicators, call print_scalarIndicators() to view any scalar indicators.
#To build a plot, call init_plot() and hand it the desired paramters. Once init_plot() has been called and any indicators you wish to plot have been added to self.indicator_dict, start adding subplots to the figure. The subplots will appear as a
#single column in the order in which you called the subplot methods. The following are the subplot methods:
#   add_subplotIndicators(): Add a subplot in which any number of indicators in self.indicator_dict are plotted. The indicator_dict option should be a list of keys of self.indicator_dict, or a dictionary of the same format as self.indicator_dict.
#                            These are the indicators which will be plotted against the simPoints. Add any extra custom lines (like e.g. the noise spectrum you are trying to attenuate, or desired frequency response of the filter) via extraPlotSeries,
#                            whose keys are the custom datasets' names, and the entries are ([a vector of y-values, same length as simPoints],"the dataset's units"). If no colors are specified, or if too few are specified, a spectrum of colors
#                            from blue to red will be autogenerated.
#   add_subplotCostFunction(): Add a subplot in which the cost function is plotted against iterations. If you wish to zoom in on a particular segment of the optimization, make grand_iterations a nonempty list to only display those grand_iterations.
#   add_subplotStructure(): Add a subplot in which the structure of the device at a given iteration OR the structure of a custom device is plotted. If both an iteration and a custom structure is specified, the iteration will take presidence.
#Once you've added all the subplots you want to add, call get_plot() to plot it. multiplot() in misc_uitls.plot_utils will then generate the figure and display/save it, per your instructions in init_plot().
    def init_analysis(self,simRange,simResolution,simScale,simType,third_variables=['N/A',],logSim=False,substrates=[],superstrates=[],num_layers=None):#Load structure/sim parameters
        self.analysis_xScale=1
        self.analysis_yScale=1
        self.num_layers_user=num_layers
        self.aid=agnostic_invDes(self.costFunction,self.costFunction_gradPhi,self.costFunction_gradE,self.globalBoundaries,self.transferMatrices,self.transferMatrices_gradPhi)#Instantiate the agnostic inverse designer
        self.logSim,self.simType,self.third_vars,self.simScale,self.substrates,self.superstrates=logSim,simType,third_variables,simScale,substrates,superstrates
        if len(simRange)>2:
            self.simPoints=np.array(simRange)
            self.simResoluion=len(simRange)
        elif logSim:#Generate the simulation points
            self.simPoints=np.logspace(np.log10(simRange[0]),np.log10(simRange[1]),simResolution)
        else:
            self.simPoints=np.linspace(simRange[0],simRange[1],simResolution)
        self.callPoints=self.simPoints*simScale#Generate the call points, in SI units, for use by material functions
        self.indicator_dict={}
        return

    def get_indicatorsEvo(self,indicators,grand_iteration,iteration,name='default'):#Get indicators from an optimziaiton iteration
        params=self.evo[grand_iteration][iteration]['parameters']
        if name=='default':
            name=str([grand_iteration,iteration])
        if self.simType=='fixedPoint':#If the simtype is fixedPoint, there is one more x than there is y
            xdiv=int((len(params)+1)/2)
        else:
            xdiv=int(len(params)/2)
        new_dict=self._get_indicators(indicators,params[:xdiv],params[xdiv:],name)
        self.indicator_dict={**self.indicator_dict,**new_dict}#Merge the new indicators dictionary into the old one, overwriting old entries if they have the same key
        return
    def get_indicatorsCustom(self,indicators,custom_x,custom_y,name):#Get indicators from a custom structure
        new_dict=self._get_indicators(indicators,custom_x,custom_y,name)
        self.indicator_dict={**self.indicator_dict,**new_dict}
        return  
    def _get_indicators(self,indicators,x,y,name):#Lower-level function to interface with physics package
        self.num_layers=len(x)
        if self.simType=='fixedPoint':
            if self.num_layers_user==None:
                self.num_layers=7
            else:
                self.num_layers=self.num_layers_user
        self.var_x,self.var_y=True,True
        self.aid.simulate(list(np.array(x)*self.analysis_xScale)+list(np.array(y)*self.analysis_yScale))#Run the simulation for the given structure
        return self.pp.indicators(indicators,self.simPoints,self.third_vars,self.simScale,self.aid.simDict['fields'],self.all_mat1params,self.all_mat2params,self.all_param0s,self.all_paramNs,self.all_custom_input,name)#Get the indicator dictionary

    def print_scalarIndicators(self):#Display any scalar indicators
        for key in self.indicator_dict.keys():
            v=self.indicator_dict[key]
            if np.isscalar(v[0]):
                print(key+': '+str(v[0])+' '+v[1])
        return
    
    def init_plot(self,save_name='test',save_dir='',save=False,show=True,size=(12,12),dpi=100,dark=False,savetype='.png',bbox_inches=None,tight_layout=True,
                    subplots_adjust={'left':None,'right':None,'top':None,'bottom':None,'hspace':None,'wspace':None},sharex=False, height_ratios=None):#Set up the global plot dictionary
        self.x_sets,self.y_sets,self.dataset_dictionaries,self.subplot_dictionaries,self.global_dictionary=[],[],[],[],{'save':save,'save_name':save_dir+save_name+savetype,'dpi':dpi,'show':show,'size':size,'bbox_inches':bbox_inches,'tight_layout':tight_layout,'subplots_adjust':subplots_adjust,'sharex':sharex,'height_ratios':height_ratios}
        self.dark=dark#dark makes all axes/text white, for quick covnersion to a dark theme for powerpoints
        return

    #indicator_dict is either a list of keys of the current indicator_dict or a seperate indicator_dict. These lines will all be plotted on the same subplot, with legend keys equal to their dictionary keys
    #extraPlotSeries is a dictionary of the same format as the indicator_dict (keys=dataset labels, entries=(vector of length=len(self.simPoints),string=units))
    def add_subplotIndicators(self,indicator_dict,axiscolor='black',tickfontcolor='black',labelfontcolor='black',labelfontsize=25,tickfontsize=15,legendfontsize=15,legendfontcolor='black',legend=False,lineColors={},simPointColors={},
                              linestyles={},linewidths={},markersizes={},markerstyles={},plot_simPoints=False,extraPlotSeries={},title=None,xlabel=None,ylabel=None,patches=[],noYticks=False,noXticks=False,xlims=None,ylims=None,xscale=1,yscale=1,
                              datasetLabels={},axiswidth=0.8,tickwidth=0.8,ticklength=3.5,legendframe=True,legendlocation='best',xticklocations=None,xticklabels=None,yticklocations=None,yticklabels=None,alphas={}):#Add subplot with plotted indicators
        if self.dark:
            axiscolor,tickfontcolor,labelfontcolor,legendfontcolor='white','white','white','white'
        self.subplot_dictionaries.append({'axiscolor':axiscolor,'tickfontcolor':tickfontcolor,'labelfontcolor':labelfontcolor,'labelfontsize':labelfontsize,
                                        'tickfontsize':tickfontsize,'legendfontsize':legendfontsize,'legendfontcolor':legendfontcolor,'legend':legend,'title':title,
                                        'x_label':xlabel,'y_label':ylabel,'logx':self.logSim,'patches':patches,'noYticks':noYticks,'noXticks':noXticks,'xlims':xlims,
                                        'ylims':ylims,'axiswidth':axiswidth,'tickwidth':tickwidth,'ticklength':ticklength,'legendframe':legendframe,'legendlocation':legendlocation,'xticklocations':xticklocations,'xticklabels':xticklabels,'yticklocations':yticklocations,'yticklabels':yticklabels})
        self.dataset_dictionaries.append([])
        self.x_sets.append([])
        self.y_sets.append([])
        extra_keys=list(extraPlotSeries.keys())
        if type(indicator_dict)==dict:#If we've been given a full indicator_dict
            ui=tuple(indicator_dict.keys())
            use_inds=list([uiv for uiv in ui if not np.isscalar(indicator_dict[uiv][0])])#Screen out scalar indicators
        else:#If we've been given a list of keys of self.indicator_dict
            use_inds=list([uiv for uiv in indicator_dict if not np.isscalar(self.indicator_dict[uiv][0])])#Screen out scalar indicators
            indicator_dict=self.indicator_dict
        lineColorKeys,simPointColorKeys,linestyleKeys,linewidthKeys,markersizeKeys,markerstyleKeys,labelKeys,alphaKeys=list(lineColors.keys()),list(simPointColors.keys()),list(linestyles.keys()),list(linewidths.keys()),list(markersizes.keys()),list(markerstyles.keys()),list(datasetLabels.keys()),list(alphas.keys())
        need_line_colors=self.get_color_gradient('#3198f5','#fb2d1e',max((len(use_inds)+len(extra_keys))-len(lineColorKeys),2))
        need_point_colors=self.get_color_gradient('#3198f5','#fb2d1e',max((len(use_inds)+len(extra_keys))-len(simPointColorKeys),2))
        for key in use_inds+extra_keys:
            if key not in lineColorKeys:
                lineColors[key]=need_line_colors.pop(0)
            if key not in simPointColorKeys:
                simPointColors[key]=need_point_colors.pop(0)
            if key not in linestyleKeys:
                linestyles[key]='-'
            if key not in linewidthKeys:
                linewidths[key]=1
            if key not in markersizeKeys:
                markersizes[key]=3
            if key not in markerstyleKeys:
                markerstyles[key]='.'
            if key not in labelKeys:
                datasetLabels[key]=key
            if key not in alphaKeys:
                alphas[key]=1
        for j in range(len(extra_keys)):#Run through each custom line. These will go under the indicator lines.
            key=extra_keys[j]
            self.dataset_dictionaries[-1].append({'type':'plot','markersize':0,'marker':'','markercolor':lineColors[key],'linewidth':linewidths[key],'linecolor':lineColors[key],'dataset_name':datasetLabels[key],'alpha':alphas[key]})#Add it to the plot dictionaries
            self.x_sets[-1].append(np.array(self.simPoints)*xscale)
            self.y_sets[-1].append(np.array(extraPlotSeries[extra_keys[j]][0])*yscale)
        for i in range(len(use_inds)):#Run through each indicator
            key=use_inds[i]
            self.dataset_dictionaries[-1].append({'type':'plot','markersize':0,'marker':'', 'markercolor':lineColors[key],'linestyle':linestyles[key],'linewidth':linewidths[key],'linecolor':lineColors[key],'dataset_name':datasetLabels[key],'alpha':alphas[key]})#Add it to the plot dictionaries
            self.x_sets[-1].append(np.array(self.simPoints)*xscale)
            y=indicator_dict[use_inds[i]][0]
            self.y_sets[-1].append(np.array(y)*yscale)
            if plot_simPoints:#If we're plotting the simulation points
                sp_y=[]
                for sp in self.evo[-1]['simPoints']:#Find the values closest to the simulation point, average them, and plot them
                    sp_si=sp*self.evo[-1]['simScale']
                    j,jp1=self.findBoundInds(self.callPoints,sp_si)
                    sp_y.append((y[j]+y[jp1])/2)
                self.dataset_dictionaries[-1].append({'type':'scatter','marker':markerstyles[key],'markersize':markersizes[key],'markercolor':simPointColors[key],'dataset_name':datasetLabels[key]+' simPoints','alpha':alphas[key]})
                self.x_sets[-1].append(np.array(self.evo[-1]['simPoints'])*xscale)
                self.y_sets[-1].append(np.array(sp_y)*yscale)
        i+=1
        return

    #type=='full' plots the full cost function, physical+regularization. type=='phys' plots the physical part. type=='reg' plots the regularization terms
    def add_subplotCostFunction(self,grand_iterations=[],axiscolor='black',tickfontcolor='black',labelfontcolor='black',labelfontsize=25,tickfontsize=15,legendfontsize=15,legendfontcolor='black',legend=False,lineColor='#0f88f5',pointColor='#cf1111',
                                linewidth=1,markersize=3,plot_grand_iters=True,title=None,xlabel='Iterations',ylabel='Cost Function',tp='full',linestyle='-',markerstyle='.',noYticks=False,noXticks=False,xlims=None,ylims=None,axiswidth=0.8,
                                tickwidth=0.8,ticklength=3.5,legendframe=True,logx=False,logy=False,legendlocation='best',xticklocations=None,xticklabels=None,yticklocations=None,yticklabels=None,alpha=1):#Add a cost function subplot
        tpd={'full':'costFunction','both':'costFunction','reg':'costFunction-Regularization','regularization':'costFunction-Regularization','phys':'costFunction-Physical','physical':'costFunction-Physical'}
        tp=tpd[tp.lower()]
        if self.dark:
            axiscolor,tickfontcolor,labelfontcolor,legendfontcolor='white','white','white','white'
        self.subplot_dictionaries.append({'axiscolor':axiscolor,'tickfontcolor':tickfontcolor,'labelfontcolor':labelfontcolor,'labelfontsize':labelfontsize,
                                        'tickfontsize':tickfontsize,'legendfontsize':legendfontsize,'legendfontcolor':legendfontcolor,'legend':legend,'title':title,
                                        'x_label':xlabel,'y_label':ylabel,'noYticks':noYticks,'noXticks':noXticks,'xlims':xlims,'ylims':ylims,'axiswidth':axiswidth,
                                        'tickwidth':tickwidth,'ticklength':ticklength,'legendframe':legendframe,'logx':logx,'logy':logy,'legendlocation':legendlocation,'xticklocations':xticklocations,'xticklabels':xticklabels,'yticklocations':yticklocations,'yticklabels':yticklabels})
        self.dataset_dictionaries.append([])
        self.x_sets.append([])
        self.y_sets.append([])
        cf_i,cf_gi,iters,gis=[],[],[],[]
        if len(grand_iterations)==0:#If we haven't been given a list of grand iterations, then we're plotting all of them
            grand_iterations=self.grand_iters
        i=0
        for gi in grand_iterations:#Run through each grand iteration
            for iter in range(len(self.evo[gi])):#Run through each iteration
                cf=self.evo[gi][iter][tp]*int(not logy)+abs(self.evo[gi][iter][tp])*int(logy)
                if iter==0:#If this is the first iteration, make note of this as the start of a new grand iteration
                    cf_gi.append(cf)
                    gis.append(i)
                cf_i.append(cf)
                iters.append(i)
                i+=1
        self.dataset_dictionaries[-1].append({'type':'plot','markersize':0,'marker':'','markercolor':lineColor,'linewidth':linewidth,'linestyle':linestyle,'linecolor':lineColor,'dataset_name':'costFunciton','alpha':alpha})#Add everthing to plot dictionaries
        self.x_sets[-1].append(iters)
        self.y_sets[-1].append(cf_i)
        if plot_grand_iters:
            self.dataset_dictionaries[-1].append({'type':'scatter','markersize':markersize,'markercolor':pointColor,'marker':markerstyle,'dataset_name':'grandIterations'})
            self.x_sets[-1].append(gis)
            self.y_sets[-1].append(cf_gi)
        return

    #grand_iteration and iteration will overrride custom_x,custom_y if they are not None
    #superstrates/substrates here are lists of three-tuples (thickness of sub/superstrate, y of sub/superstrate, color of sub/superstrate)
    def add_subplotStructure(self,grand_iteration=None,iteration=None,custom_x=None,custom_y=None,axiscolor='black',tickfontcolor='black',labelfontcolor='black',labelfontsize=25,tickfontsize=15,color1='#a684ff',color2='#cc064c',
                              title=None,xlabel=None,ylabel=None,xscale=1,superstrates=[],substrates=[],linewidth=1,markersize=3,linestyle='-',markerstyle='.',noYticks=False,noXticks=False,xlims=None,ylims=None,param0=0,paramN=0,yscale=1,
                              incident_pad=[0,1,'#e9f3fb'],transmission_pad=[0,1,'#e9f3fb'],axiswidth=0.8,tickwidth=0.8,ticklength=3.5,legendframe=True,legendlocation='best',xticklocations=None,xticklabels=None,yticklocations=None,yticklabels=None):#Add subplot of structures
        if grand_iteration==None and iteration==None and custom_x==None and custom_y==None:#If the user messed up, tell them. Doesn't catche everthing that could go wrong, but probably the most common one.
            print('You appear to have attempted to plot nothing. Please ensure EITHER grand_iteration and iteration are both NOT None OR custom_x and custom_y are both NOT None.')
            return
        if self.dark:
            axiscolor,tickfontcolor,labelfontcolor,legendfontcolor='white','white','white','white'
        color1_sequence,color2_sequence,struc_x,struc_y=[],[],[],[]
        if incident_pad[0]!=0:
            struc_x.append(incident_pad[0]*xscale)
            struc_y.append(incident_pad[1])
            color1_sequence.append(incident_pad[2])
            color2_sequence.append(incident_pad[2])
        for ss in superstrates:#Run through the superstrates, add them to the structure
            color1_sequence.append(ss[2])
            color2_sequence.append(ss[2])
            struc_x.append(ss[0]*xscale)
            struc_y.append(ss[1])
        if grand_iteration!=None and iteration!=None:#If we're plotting a structure from evo, grab its x/y parameters and add them to the structure lists
            params=self.evo[grand_iteration][iteration]['parameters']
            xdiv=int((len(params)+int(self.simType=='fixedPoint'))/2)
            struc_x+=list(np.array(params[:xdiv])*xscale)
            struc_y+=list(np.array(params[xdiv:])*yscale)
            color1_sequence+=[color1]*len(params[:xdiv])
            color2_sequence+=[color2]*len(params[:xdiv])
        else:#If we're plotting a custom structure, add it to the structure list
            struc_x+=list(np.array(custom_x)*xscale)
            struc_y+=custom_y
            color1_sequence+=[color1]*len(custom_x)
            color2_sequence+=[color2]*len(custom_x)
        for ss in substrates:#Dido for substrates
            color1_sequence.append(ss[2])
            color2_sequence.append(ss[2])
            struc_x.append(ss[0]*xscale)
            struc_y.append(ss[1])
        if transmission_pad[0]!=0:
            struc_x.append(transmission_pad[0]*xscale)
            struc_y.append(transmission_pad[1])
            color1_sequence.append(transmission_pad[2])
            color2_sequence.append(transmission_pad[2])
        if self.simType=='independent':#If this is an independent simulation
            pts=[]#The layers will be rectangles drawn on a set of empty axes
            running_x=-incident_pad[0]*xscale
            for i in range(len(struc_x)):#Create the rectangles
                if struc_y[i]>0:
                    pts.append(patches.Rectangle((running_x,0),struc_x[i],struc_y[i],color=color2_sequence[i]))
                if struc_y[i]<1:
                    pts.append(patches.Rectangle((running_x,struc_y[i]),struc_x[i],1-struc_y[i],color=color1_sequence[i]))
                running_x+=struc_x[i]
            xMax,yMax,xs,ys=sum(struc_x),max(struc_y),[],[]#The plot values will be empty, spawning a set of empty axes
        else:#If this is a fixedPoint simulation
            xs,ys=[0],[param0*yscale]#Then the plot values will be the fixed points (and sub/superstrates). The first one is at (0,param0)
            running_x=0
            for i in range(len(struc_x)-1):#Run through each x, except the last one which does not have an associated y
                running_x+=struc_x[i]
                xs.append(running_x)
                ys.append(struc_y[i])
            running_x+=struc_x[-1]
            xs.append(running_x)#The last fixed point is at (xMax,paramN)
            ys.append(paramN*yscale)
            pts,xMax,yMax=[],max(xs),max(ys)#We have no patches in this case
        if xlims==None:
            xlims=(-0.05*xMax,xMax*1.05)
        if ylims==None:
            ylims=(0,yMax)
        self.subplot_dictionaries.append({'axiscolor':axiscolor,'tickfontcolor':tickfontcolor,'labelfontcolor':labelfontcolor,'labelfontsize':labelfontsize,
                                        'tickfontsize':tickfontsize,'legend':False,'title':title,'x_label':xlabel,'y_label':ylabel,'xlims':xlims,'ylims':ylims,
                                        'patches':pts,'noYticks':noYticks,'noXticks':noXticks,'axiswidth':axiswidth,'tickwidth':tickwidth,'ticklength':ticklength,'legendframe':legendframe,'legendlocation':legendlocation,'xticklocations':xticklocations,'xticklabels':xticklabels,'yticklocations':yticklocations,'yticklabels':yticklabels})
        self.dataset_dictionaries.append([{'type':'plot','markersize':markersize,'markercolor':color2,'linewidth':linewidth,'linecolor':color1,'linestyle':linestyle,'marker':markerstyle,'dataset_name':"structure"}])
        self.x_sets.append([xs])
        self.y_sets.append([ys])
        return

    def add_subplotFields(self,grand_iteration=None,iteration=None,custom_x=None,custom_y=None,xscale=1,yscale=1,resolution=500,leftPad=0.1,rightPad=0.1,colormap='magma',physicsPackageArg=None,axiscolor='black',tickfontcolor='black',
                            labelfontcolor='black',labelfontsize=25,tickfontsize=15,title=None,xlabel=None,ylabel=None,noYticks=False,noXticks=False,xlims=None,ylims=None,patches=[],colorbar=True,zlims=None,colorbarLabel='',logx=False,
                            logy=False,colorbarNorm='linear',ysim=None,eliminate_incident=False,axiswidth=0.8,tickwidth=0.8,ticklength=3.5,legendframe=True,contourf=False,contourlevels=50,legendlocation='best',xticklocations=None,xticklabels=None,yticklocations=None,yticklabels=None):
        self.var_x,self.var_y=True,True
        if grand_iteration is not None and iteration is not None:
            params=self.evo[grand_iteration][iteration]['parameters']
        else:
            params=list(custom_x)+list(custom_y)
        xs=params[:int((len(params)+int(self.simType=='fixedPoint'))/2)]
        self.num_layers=int(len(params))/2
        if self.simType=='fixedPoint':
            if self.num_layers_user==None:
                self.num_layers=7
            else:
                self.num_layers=self.num_layers_user
        self.aid.simulate(params)
        subt,supt=list([ss['thickness'] for ss in self.substrates]), list([ss['thickness'] for ss in self.superstrates])
        totx,totsup,totsub,num_sub,num_x,num_sup=sum(xs),sum(supt),sum(subt),len(self.substrates),len(xs),len(self.superstrates)
        tot_x_unpadded=sum(xs)+sum(supt)+sum(subt)
        tot_x=tot_x_unpadded*(1+leftPad+rightPad)
        xsim=np.linspace(-1*tot_x_unpadded*leftPad,tot_x_unpadded*(1+rightPad),resolution)
        z_mesh,x_mesh,y_mesh=[],[],[]
        i=0
        zmax,zmin=-np.inf,np.inf
        for tv in self.third_vars:
            j=0
            for sp in self.simPoints:
                self.sim_params['third_vars'],self.sim_params['simPoint'],self.sim_params['callPoint']=tv,sp,self.callPoints[j]
                z_mesh.append(np.array(self.interpolate_fields(self.aid.simDict['fields'][i],self.xLayer,self.yLayer,xsim,totx,totsup,totsub,num_sup,num_sub,num_x,supt,subt,physicsPackageArg,eliminate_incident)))
                zmax,zmin=max(zmax,max(z_mesh[-1])),min(zmin,min(z_mesh[-1]))
                x_mesh.append(xsim*xscale)
                try:
                    y_mesh.append(np.array([ysim[i]]*len(xsim))*yscale)
                except:
                    y_mesh.append(np.array([sp]*len(xsim))*yscale)
                j+=1
                i+=1
        self.x_sets.append([z_mesh])
        self.y_sets.append([(x_mesh,y_mesh)])
        if zlims==None:
            zlims=[None,None]
        else:
            if zlims[0]==None:
                zlims[0]=zmin
            if zlims[1]==None:
                zlims[1]=zmax
        self.subplot_dictionaries.append({'axiscolor':axiscolor,'tickfontcolor':tickfontcolor,'labelfontcolor':labelfontcolor,'labelfontsize':labelfontsize,
                                        'tickfontsize':tickfontsize,'legend':False,'title':title,'x_label':xlabel,'y_label':ylabel,'xlims':xlims,'ylims':ylims,
                                        'patches':patches,'noYticks':noYticks,'noXticks':noXticks,'colorbar':colorbar,'zlims':zlims,'colorbarLabel':colorbarLabel,
                                        'logx':logx,'logy':logy,'colorbarNorm':colorbarNorm,'axiswidth':axiswidth,'tickwidth':tickwidth,'ticklength':ticklength,
                                        'legendframe':legendframe,'contourlevels':contourlevels,'legendlocation':legendlocation,'xticklocations':xticklocations,'xticklabels':xticklabels,'yticklocations':yticklocations,'yticklabels':yticklabels})
        if contourf:
            self.dataset_dictionaries.append([{'type':'contourf','colormap':colormap,'dataset_name':"fields"}])
        else:
            self.dataset_dictionaries.append([{'type':'pcolormesh','colormap':colormap,'dataset_name':"fields"}])
        return
            
        
    def interpolate_fields(self,fields,xs,ys,xsim,totx,totsup,totsub,num_sup,num_sub,num_x,supt,subt,physicsPackageArg,eliminate_incident):
        rf=[]
        for x in xsim:
            #print('x: '+str(x))
            if x<=0:
                if eliminate_incident:
                    use_fields=np.array((0,fields[0][1]))
                else:
                    use_fields=fields[0]
                rf.append(self.pp.interpolate(use_fields,abs(x),1,self.param0Call(self.sim_params),self.param0Call(self.sim_params),self.sim_params,reverse=True,physicsPackageArg=physicsPackageArg))
                #print('Incident; svd: '+str(abs(x))+', fields ind: '+str(0)+', y: '+str(1)+', interp_value: '+str(rf[-1]))
            elif x<totsup:
                rsup=0
                for i in range(len(supt)):
                    rsup+=supt[i]
                    if x<rsup:
                        rf.append(self.pp.interpolate(fields[i+1],x-rsup+supt[i],1,self.superstrates[i]['material_call'](self.sim_params),self.superstrates[i]['material_call'](self.sim_params),self.sim_params,physicsPackageArg=physicsPackageArg))
                        #print('super; svd: '+str(x-rsup+supt[i])+', fields ind: '+str(i+1)+', y: '+str(1)+', interp_value: '+str(rf[-1]))
                        break
            elif x<totx+totsup:
                rx=totsup
                for i in range(len(xs)):
                    rx+=xs[i]
                    if x<rx:
                        rf.append(self.pp.interpolate(fields[i+1+num_sup],x-rx+xs[i],ys[i],self.mat1Call(self.sim_params),self.mat2Call(self.sim_params),self.sim_params,physicsPackageArg=physicsPackageArg))
                        #print('layer; svd: '+str(x-rx+xs[i])+', fields ind: '+str(i+1+num_sup)+', y: '+str(ys[i])+', interp_value: '+str(rf[-1]))
                        break
            elif x<totsup+totx+totsub:
                rsub=totsup+totx
                for i in range(len(subt)):
                    rsub+=subt[i]
                    if x<rsub:
                        rf.append(self.pp.interpolate(fields[i+1+num_sup+num_x],x-rsub+subt[i],1,self.substrates[i]['material_call'](self.sim_params),self.substrates[i]['material_call'](self.sim_params),self.sim_params,physicsPackageArg=physicsPackageArg))
                        #print('sub; svd: '+str(x-rsub+subt[i])+', fields ind: '+str(i+1+num_sup+num_x)+', y: '+str(1)+', interp_value: '+str(rf[-1]))
                        break
            else:
                rf.append(self.pp.interpolate(fields[-1],x-totsup-totx-totsub,1,self.paramNCall(self.sim_params),self.paramNCall(self.sim_params),self.sim_params,physicsPackageArg=physicsPackageArg))
                #print('trans; svd: '+str(x-totsup-totx-totsub)+', fields ind: '+str(-1)+', y: '+str(1)+', interp_value: '+str(rf[-1]))
        return rf
        
    def get_plot(self):#Generate the plot
        self.multiplot(self.x_sets,self.y_sets,self.dataset_dictionaries,self.subplot_dictionaries,self.global_dictionary)
        return
