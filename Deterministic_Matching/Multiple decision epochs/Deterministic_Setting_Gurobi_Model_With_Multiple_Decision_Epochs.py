import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt 
import networkx as nx

global t; global a; global b; global driverindex; global orderindex;
    
t = 0; a = []; 
b = [{'index':0, 'assigned':False, 'sa': 0, 'ma':0, 'tam':None, 'tmin':None, 'tmax':None}]; #the first entry is for null order
driverindex = 0; orderindex = 1; #driver index is 0-based, order is 1-based as 0 is None

def generate_inst(): 
    global driverindex; global orderindex;
    numdrivers = np.random.randint(1,3); #5<=numdrivers<=10 
    numorders = np.random.randint(3,6); 
    delta = 5; #defines maximum time an order can be fulfilled 
    
    loc_set = [[0.5 + x, 0.5 + y] for x in range(10) for y in range(10)]
    
    def closest_node(node):
        nodes = np.asarray(loc_set)
        closest_ind = distance.cdist([node], nodes).argmin() 
        return loc_set[closest_ind]
    
    #print('closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)]): ', closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)]))
    
    newdrivers = [{'sa':1, 'ma':1, 'loc':closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)]), 'tam':t, 'index':driverindex+i} for i in range(numdrivers)]  
    
    print('The following new drivers enter at this epoch: ', [elt['index'] for elt in newdrivers])
    
    a.extend(newdrivers) 
    
    driverindex = a[-1]['index']+1 #next driver must have index 1 greater 
    
    #a = (sa, ma, oa, tam); tam (time driver will be available) = t (driver is available now) 
     
    #minrevenue = 1; maxrevenue = 11 #if revenue is small more drivers are unassigned 
    minrevenue = 11; maxrevenue = 21 #revenue is random integer x where minrevenue<=x<=maxrevenue 
    
    orderindex = b[-1]['index']+1 
    neworders = []
    for i in range(numorders):
        o_discrete = closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)])
        d_discrete = closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)])

        while o_discrete == d_discrete:  # to make sure origin != dest
            d_discrete = closest_node([np.random.uniform(0, 10), np.random.uniform(0, 10)])

        b.append({'ob':o_discrete, 'db':d_discrete, 'tmin':t+1, 'tmax':t+delta, 'revenue':np.random.randint(minrevenue, maxrevenue+1), 'assigned':False, 'index':orderindex}) 
        neworders.append(b[-1]['index'])
        
        orderindex+=1
        
    print('\nOrder information \n')
    print('The following new orders are placed at this epoch: ', neworders) 
    
def dist(vec1, vec2):
    #Euclidean distance between vec1 and vec2 
    return ((vec1[0]-vec2[0])**2 + (vec1[1]-vec2[1])**2)**0.5 

def c(i, j):  
    #cost for driver i to deliver order j; proportional to Euclidean distance from driver to order origin plus Euclidean distance from order origin to order destination  
    #i is an index and not an attribute vector, as is j
    return dist(a[i]['loc'], b[j]['ob']) + dist(b[j]['ob'], b[j]['db'])

def graph(a, b, assignmentdrivertoorder): 
    B = nx.Graph() #Initialize the graph
    top_nodes = ['Driver ' + str(a[i]['index']) + ' sa: ' + str(a[i]['sa']) + ' ma: ' + str(a[i]['ma']) + ' tam: ' + str(a[i]['tam']) for (i, j) in assignmentdrivertoorder][::-1] #only considering active and available drivers
    bottom_nodes = ['Order ' + str(b[j]['index']) + ' tmin: ' + str(b[j]['tmin']) + ' tmax: ' + str(b[j]['tmax']) for (i, j) in assignmentdrivertoorder][::-1]
    B.add_nodes_from(top_nodes, bipartite=0)
    B.add_nodes_from(bottom_nodes, bipartite=1)
     
    # Add edges between drivers and matched orders
    edges = [('Driver ' + str(a[i]['index'])+ ' sa: ' + str(a[i]['sa']) + ' ma: ' + str(a[i]['ma']) + ' tam: ' + str(a[i]['tam']), 'Order ' + str(b[j]['index']) + ' tmin: ' + str(b[j]['tmin']) + ' tmax: ' + str(b[j]['tmax']) ) for (i, j) in assignmentdrivertoorder]
    B.add_edges_from(edges)

    plt.figure(figsize=(20, 6))  # Adjust the figure size for better readability 

    edges = B.edges()

    pos = dict()
    pos.update((n, (1, i * 2)) for i, n in enumerate(top_nodes))  #i * 2 to Increase the spacing between nodes in X-axis
    pos.update((n, (2, i * 2)) for i, n in enumerate(bottom_nodes))  # Increase the spacing between nodes in X-axis

    nx.draw_networkx(B, pos=pos, node_size=200, node_color='skyblue', edge_color='gray')  # Customize the node and edge appearance

    plt.axis('off')  # Remove the axis

    plt.show()  
    
def optimize(a, b):
    
    speed = 3; delta = 5; #assumed same for all drivers  
    
    numdrivers = len(a)
    numorders = len(b) #numorders is 1 more than actual number of orders because first element of b is None to signify a driver is not assigned to an order  
     
    m = gp.Model('driverallocation');  m.Params.LogToConsole = 0
    
    assign = m.addVars([(i,j) for i in range(numdrivers) for j in range(numorders)], name = 'assign', vtype = GRB.BINARY) 
    #assign[i,j] means driver i (0-based index) is assigned to order j (1-based index; if j = 0 it indicates not being assigned to an order)
    
    m.addConstrs(gp.quicksum(assign[i,j] for j in range(numorders)) == 1 for i in range(numdrivers))  
    #each driver is assigned at most once (since a is an index not a vector Ra = 1) or not assigned at all 
    #constraint (4b); since each driver has a unique index, Rta is 1 

    m.addConstrs(gp.quicksum(assign[i,j] for i in range(numdrivers)) <= 1 for j in range(1, numorders)) 
    #each order (except None, which represents not being assigned) is fulfilled at most once 
    #constraint (4c); here j ranges from 1 to numorders because the possibility of unassigned orders is not relevant 
      
    m.addConstrs(assign[i,j] <= min(a[i]['sa'], a[i]['ma']) for j in range(1, numorders) for i in range(numdrivers))
    #the driver must be both active and available to be assigned; if the driver is inactive or unavailable assign[i,j] = 0  
    #To avoid the 'Infeasible model' error j must range from 1 to numorders not 0, otherwise assign[i,0] can be forced to be 0 if the driver is unavailable/inactive
         
    m.addConstrs(assign[i,j] <= int(int(np.ceil(c(i, j)/speed)) <= delta) for j in range(1, numorders) for i in range(numdrivers))
    #int(np.ceil(c(i, j)/speed)) = number of epochs for driver to fulfill the order 
    #assign[i,j] must be 0 if the order can't be fulfilled by tmax; delta = 5 so tmax = tcurr+5 always  
    
    m.addConstrs(assign[i,j] <= int(a[i]['tam'] <= t) for j in range(1, numorders) for i in range(numdrivers)) 
    #assign[i,j] must be 0 if the driver is not available for matching  
    
    m.addConstrs(assign[i,j] <= (1-int(b[j]['assigned'])) for j in range(1, numorders) for i in range(numdrivers)) 
    #assign[i,j] must be 0 if the order is assigned - that is, b[i]['assigned'] = 1 
      
    m.setObjective(gp.quicksum((b[j]['revenue'] - c(i, j))*assign[i,j] for i in range(numdrivers) for j in range(1, numorders)), GRB.MAXIMIZE)  
    #j ranges from 1 to numorders because if j = 0 no revenue is earned as a driver was not assigned     
        
    m.optimize()  
     
    #UPDATE PARAMETERS
    assignmentdrivertoorder = [(i, j) for i in range(numdrivers) for j in range(numorders) if assign[i, j].x == 1 and a[i]['tam'] <= t]
    #this ensures drivers and orders that are assigned in this epoch only are considered, and not those that were previously assigned or not assigned
    
    bnew = [None]
    for (i,j) in assignmentdrivertoorder: #update the parameters a and b 
        if j > 0: #do not consider null assignment 
            tijfulfill = int(np.ceil(c(i, j)/speed)) #number of epochs for driver to fulfill the order    
            a[i]['tam'] = t + tijfulfill
            if tijfulfill > 1: 
                a[i]['ma'] = 0 #the driver is not available for matching, as they are fulfilling an order  
                #but if tijfulfill = 1 then the driver is available at next epoch  
            a[i]['loc'] = b[j]['db'] #driver will reach the order's destination once the order is fulfilled     
            b[j]['assigned'] = True #order is assigned to a driver 
            
    for j in range(1, len(b)): #assign values to bnew, the updated b vector 
        if b[j]['tmax'] > t+1 and b[j]['assigned'] == False: #unassigned orders with tmax <= t+1 must be dropped; it takes at least 1 epoch to fulfill 
            bnew.append(b[j]) 
    
    #DISPLAY ORDER INFORMATION 
    print('The following orders are left unassigned at this epoch: ', [b[j]['index'] for j in range(1, len(b)) if b[j]['assigned'] == False])
    print('The following orders were previously assigned and are being fulfilled at this epoch: ', [b[j]['index'] for j in range(1, len(b)) if b[j]['assigned'] == True and b[j]['tmin'] <= t])
    print('The following orders are assigned at this epoch: ', [b[j]['index'] for j in range(1, len(b)) if b[j]['assigned'] == True and b[j]['tmin'] == t+1])    
    print('The following unassigned orders are dropped at this epoch because their time limit is exceeded: ', [b[j]['index'] for j in range(1, len(b)) if b[j]['assigned'] == False and b[j]['tmax'] == t])
    
    #print('b: ', b)  
    
    #NETWORK DIAGRAM 
    graph(a, b, assignmentdrivertoorder) 
       
    b = bnew   

print('DECISION EPOCH: ', t)

generate_inst()
    
optimize(a, b)    

for t in range(1, 10): 
    
    print('\nDECISION EPOCH: ', t); print()
    
    for i in range(len(a)): 
        if a[i]['tam'] <= t: 
            a[i]['ma'] = 1 #driver is available now, since the order is fulfilled 
   
    driversexit = True #set to False to simulate scenario where no drivers exit the system 
    
    if driversexit: 
        numexit = np.random.randint(2,5) #number of drivers that exit 
             
        exitingdriverindices = []
        pexit = 0.4; #probability an unassigned driver exits 
        anew = [];  
        
        for i in range(len(a)): 
            driverexits = bool(int(np.random.binomial(size=1, n=1, p= pexit))) 
            if a[i]['ma'] == 0 or not driverexits: 
                anew.append(a[i])
            else: 
                exitingdriverindices.append(i)  
        
        print('\nDriver information \n')
        print('The following drivers fulfil their order at this epoch: ', [a[i]['index'] for i in range(len(a)) if a[i]['tam'] == t])
        print('The following drivers are unassigned at this epoch and did not just fulfil their order: ', [a[i]['index'] for i in range(len(a)) if a[i]['tam'] < t])      
        print('The following unassigned drivers exit at this epoch: ', [a[i]['index'] for i in exitingdriverindices])
        print('The following drivers are fulfilling their order at this epoch: ', [a[i]['index'] for i in range(len(a)) if a[i]['tam'] > t and a[i]['ma'] == 0])
         
        a = anew
          
    generate_inst()

    optimize(a, b)     

    
