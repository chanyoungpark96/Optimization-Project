#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gurobipy as gp
import pandas as pd


# In[2]:


# Read the data from Excel files
data1 = np.array(pd.read_excel("SuperChipData.xlsx",sheet_name=0))
data2 = np.array(pd.read_excel("SuperChipData.xlsx",sheet_name=1))
data3 = np.array(pd.read_excel("SuperChipData.xlsx",sheet_name=2))
data4 = np.array(pd.read_excel("SuperChipData.xlsx",sheet_name=3))


# In[3]:


# Current policy shares of each facility
share1 = data1[0,1]/sum(data1[:,1]);
share2 = data1[1,1]/sum(data1[:,1]);
share3 = data1[2,1]/sum(data1[:,1]);
share4 = data1[3,1]/sum(data1[:,1]);
share5 = data1[4,1]/sum(data1[:,1]);


# In[4]:


# Define the problem data
n = 30*23
m = 5*len(data2)


# In[5]:


# Left-hand side inequalities
Aineq = np.concatenate((np.hstack((np.ones((1,n)), np.zeros((1,4*n)))),
                        np.hstack((np.zeros((1,n)), np.ones((1,n)), np.zeros((1,3*n)))),
                        np.hstack((np.zeros((1,2*n)), np.ones((1,n)), np.zeros((1,2*n)))),
                        np.hstack((np.zeros((1,3*n)), np.ones((1,n)), np.zeros((1,n)))),
                        np.hstack((np.zeros((1,4*n)), np.ones((1,n)))),
                        -np.eye(m)), axis=0)


# In[6]:


# Right-hand side inequalities
bineq = np.concatenate((data1[:,1]*1000, np.zeros((m,))), axis=0)


# In[7]:


# Left-hand side equalities
Aeq = np.zeros((len(data2), m))


# In[8]:


for i in range(len(data2)):
    for j in range(5):
        Aeq[i, i+j*n] = 1


# In[9]:


# Right-hand side equalities
beq = data2[:,2]*1000


# In[10]:


# Cost vector
c = np.zeros((m,))

# Iterate through the cost vectors
for k in range(1,6):
    for i in range(23):
        for j in range(30):
            c[(k-1)*n + i*30 + j] = c[(k-1)*n + i*30 + j] + data4[(k-1)*30 + j, 2] + data3[i*150 + j + (k-1)*30, 3]


# In[11]:


# Cost with current policy
COST1 = 0
COST2 = 0
COST3 = 0
COST4 = 0
COST5 = 0

# Sum up all of the elements to obtain the cost with current policy
for k in range(1, 6):
    for i in range(1, 24):
        for j in range(1, 31):
            if k == 1:
                COST1 += c[30*23*(k-1) + (i-1)*30 + j-1]*share1*data2[(i-1)*30+j-1,2]*1000
            elif k == 2:
                COST2 += c[30*23*(k-1) + (i-1)*30 + j-1]*share2*data2[(i-1)*30+j-1,2]*1000
            elif k == 3:
                COST3 += c[30*23*(k-1) + (i-1)*30 + j-1]*share3*data2[(i-1)*30+j-1,2]*1000
            elif k == 4:
                COST4 += c[30*23*(k-1) + (i-1)*30 + j-1]*share4*data2[(i-1)*30+j-1,2]*1000
            elif k == 5:
                COST5 += c[30*23*(k-1) + (i-1)*30 + j-1]*share5*data2[(i-1)*30+j-1,2]*1000


# In[12]:


for i in range(5):
    if COST1 == min(COST1,COST2,COST3,COST4,COST5):
        minindex = i
        print("Facility with lowest cost with current policy is Alexandria, with the cost of:"+str(COST1))
    elif COST2 == min(COST1,COST2,COST3,COST4,COST5):
        minindex = i
        print("Facility with lowest cost with current policy is Richmond, with the cost of:"+str(COST2))
    elif COST3 == min(COST1,COST2,COST3,COST4,COST5):
        minindex = i
        print("Facility with lowest cost with current policy is Norfolk, with the cost of:"+str(COST3))
    elif COST4 == min(COST1,COST2,COST3,COST4,COST5):
        minindex = i
        print("Facility with lowest cost with current policy is Roanoke, with the cost of:"+str(COST4))
    elif COST5 == min(COST1,COST2,COST3,COST4,COST5):
        minindex = i
        print("Facility with lowest cost with current policy is Charolottesville, with the cost of:"+str(COST5))


# In[13]:


# Total cost of all facilities
COST = COST1 + COST2 + COST3 + COST4 + COST5

print("Cost with current policy is:"+str(COST)) 


# In[14]:


# Create a Gurobi model
model = gp.Model()


# In[15]:


# Define the decision variables
x = model.addMVar(shape=m, lb=-1e30, ub=gp.GRB.INFINITY, name="x")


# In[16]:


# Set the objective function
model.setObjective(c @ x, sense=gp.GRB.MINIMIZE)


# In[17]:


# Add the constraints
model.addConstr(Aeq @ x == beq, name="eq")
model.addConstr(Aineq @ x <= bineq, name="ineq")


# In[18]:


# Solve the model
model.optimize()


# In[19]:


# Get the optimal solution
x_opt1 = model.getAttr('x')


# In[20]:


print("Optimal solution:")
print(x_opt1)


# In[21]:


# Preallocation
aux = 0
share = np.zeros((5, 30))


# In[22]:


# Cost of the new policy
print('Cost of the new policy is:')
print(np.dot(c, x_opt1))


# In[23]:


print("The potential save if adopting new policy:") 
print(COST-np.matmul(np.transpose(c),x_opt1))


# In[24]:


# Finding the share of each facility for each type of chip
for k in range(1, 6):
    for i in range(1, 31):
        for j in range(i, len(data2)+1, 30):
            share[k-1, i-1] += x_opt1[(k-1)*30*23 + j-1]
            aux += data2[j-1, 2]*1000
        share[k-1, i-1] /= aux
        aux = 0


# In[25]:


# Cost per facility with optimized policy
costperfac = np.zeros(5)


# In[26]:


for i in range(1, 6):
    costperfac[i-1] = np.dot(c[(i-1)*30*23:i*30*23], x_opt1[(i-1)*30*23:i*30*23])


# In[27]:


# Solve for 10% increase in production
# Right-hand side equalities
beq = data2[:,2]*1000*1.10


# In[28]:


# Cost vector
c = np.zeros((m,))


# In[29]:


# Iterate through the cost vectors
for k in range(1,6):
    for i in range(23):
        for j in range(30):
            c[(k-1)*n + i*30 + j] = c[(k-1)*n + i*30 + j] + data4[(k-1)*30 + j, 2] + data3[i*150 + j + (k-1)*30, 3]


# In[30]:


# Create a Gurobi model
model = gp.Model()


# In[31]:


# Define the decision variables
x = model.addMVar(shape=m, lb=-1e30, ub=gp.GRB.INFINITY, name="x")


# In[32]:


# Set the objective function
model.setObjective(c @ x, sense=gp.GRB.MINIMIZE)


# In[33]:


model.addConstr(Aeq @ x == beq, name="eq")
model.addConstr(Aineq @ x <= bineq, name="ineq")


# In[34]:


# Solve the model
model.optimize()


# In[35]:


# Get the optimal solution for 10% increase
x_opt2 = model.getAttr('x')


# In[36]:


print("Optimal solution for 10% increase in production:")
print(x_opt2)


# In[37]:


print("The associated costs for filling new demand is:")
print(np.matmul(np.transpose(c),x_opt2)-np.matmul(np.transpose(c),x_opt1))


# In[38]:


# Right-hand side equalities
beq = data2[:,2]*1000


# In[39]:


# Cost vector - here m instead of 5*23*30
c = np.zeros((m,));

# Preallocation of the solution vector for a change in 
x = np.zeros((3450,5));
cost = np.zeros((1,5));


# In[40]:


# Iterate through the cost vectors
for k in range(1,6):
    for i in range(23):
        for j in range(30):
            if k == 1:
                c[(k-1)*n + i*30 + j] = c[(k-1)*n + i*30 + j] + 0.85*data4[(k-1)*30 + j, 2] + data3[i*150 + j + (k-1)*30, 3]
            else:
                c[(k-1)*n + i*30 + j] = c[(k-1)*n + i*30 + j] + data4[(k-1)*30 + j, 2] + data3[i*150 + j + (k-1)*30, 3]


# In[41]:


# Create a Gurobi model
model = gp.Model()


# In[42]:


# Define the decision variables
x0 = model.addMVar(shape=m, lb=-1e30, ub=gp.GRB.INFINITY, name="x")


# In[43]:


# Set the objective function
model.setObjective(c @ x0, sense=gp.GRB.MINIMIZE)


# In[44]:


# Add the constraints
model.addConstr(Aeq @ x0 == beq, name="eq")
model.addConstr(Aineq @ x0 <= bineq, name="ineq")


# In[45]:


# Solve the model
model.optimize()


# In[46]:


# Get the optimal solution
x[:, 0] = model.getAttr('x')
cost[:, 0] = np.matmul(np.transpose(c),x[:, 0])


# In[47]:


# Iterate through the cost vectors
for k in range(1,6):
    for i in range(23):
        for j in range(30):
            if k == 2:
                c[(k-1)*n + i*30 + j] = c[(k-1)*n + i*30 + j] + 0.85*data4[(k-1)*30 + j, 2] + data3[i*150 + j + (k-1)*30, 3]
            else:
                c[(k-1)*n + i*30 + j] = c[(k-1)*n + i*30 + j] + data4[(k-1)*30 + j, 2] + data3[i*150 + j + (k-1)*30, 3]


# In[48]:


# Create a Gurobi model
model = gp.Model()


# In[49]:


# Define the decision variables
x1 = model.addMVar(shape=m, lb=-1e30, ub=gp.GRB.INFINITY, name="x")


# In[50]:


# Set the objective function
model.setObjective(c @ x1, sense=gp.GRB.MINIMIZE)


# In[51]:


# Add the constraints
model.addConstr(Aeq @ x1 == beq, name="eq")
model.addConstr(Aineq @ x1 <= bineq, name="ineq")


# In[52]:


# Solve the model
model.optimize()


# In[53]:


# Get the optimal solution
x[:, 1] = model.getAttr('x')
cost[:, 1] = np.matmul(np.transpose(c),x[:, 1])


# In[54]:


# Iterate through the cost vectors
for k in range(1,6):
    for i in range(23):
        for j in range(30):
            if k == 3:
                c[(k-1)*n + i*30 + j] = c[(k-1)*n + i*30 + j] + 0.85*data4[(k-1)*30 + j, 2] + data3[i*150 + j + (k-1)*30, 3]
            else:
                c[(k-1)*n + i*30 + j] = c[(k-1)*n + i*30 + j] + data4[(k-1)*30 + j, 2] + data3[i*150 + j + (k-1)*30, 3]


# In[55]:


# Create a Gurobi model
model = gp.Model()


# In[56]:


# Define the decision variables
x2 = model.addMVar(shape=m, lb=-1e30, ub=gp.GRB.INFINITY, name="x")


# In[57]:


# Set the objective function
model.setObjective(c @ x2, sense=gp.GRB.MINIMIZE)


# In[58]:


# Add the constraints
model.addConstr(Aeq @ x2 == beq, name="eq")
model.addConstr(Aineq @ x2 <= bineq, name="ineq")


# In[59]:


# Solve the model
model.optimize()


# In[60]:


# Get the optimal solution
x[:, 2] = model.getAttr('x')
cost[:, 2] = np.matmul(np.transpose(c),x[:, 2])


# In[61]:


# Iterate through the cost vectors
for k in range(1,6):
    for i in range(23):
        for j in range(30):
            if k == 4:
                c[(k-1)*n + i*30 + j] = c[(k-1)*n + i*30 + j] + 0.85*data4[(k-1)*30 + j, 2] + data3[i*150 + j + (k-1)*30, 3]
            else:
                c[(k-1)*n + i*30 + j] = c[(k-1)*n + i*30 + j] + data4[(k-1)*30 + j, 2] + data3[i*150 + j + (k-1)*30, 3]


# In[62]:


# Create a Gurobi model
model = gp.Model()


# In[63]:


# Define the decision variables
x3 = model.addMVar(shape=m, lb=-1e30, ub=gp.GRB.INFINITY, name="x")


# In[64]:


# Set the objective function
model.setObjective(c @ x3, sense=gp.GRB.MINIMIZE)


# In[65]:


# Add the constraints
model.addConstr(Aeq @ x3 == beq, name="eq")
model.addConstr(Aineq @ x3 <= bineq, name="ineq")


# In[66]:


# Solve the model
model.optimize()


# In[67]:


# Get the optimal solution
x[:, 3] = model.getAttr('x')
cost[:, 3] = np.matmul(np.transpose(c),x[:, 4])


# In[68]:


# Iterate through the cost vectors
for k in range(1,6):
    for i in range(23):
        for j in range(30):
            if k == 5:
                c[(k-1)*n + i*30 + j] = c[(k-1)*n + i*30 + j] + 0.85*data4[(k-1)*30 + j, 2] + data3[i*150 + j + (k-1)*30, 3]
            else:
                c[(k-1)*n + i*30 + j] = c[(k-1)*n + i*30 + j] + data4[(k-1)*30 + j, 2] + data3[i*150 + j + (k-1)*30, 3]


# In[69]:


# Create a Gurobi model
model = gp.Model()


# In[70]:


# Define the decision variables
x4 = model.addMVar(shape=m, lb=-1e30, ub=gp.GRB.INFINITY, name="x")


# In[71]:


# Set the objective function
model.setObjective(c @ x4, sense=gp.GRB.MINIMIZE)


# In[72]:


# Add the constraints
model.addConstr(Aeq @ x4 == beq, name="eq")
model.addConstr(Aineq @ x4 <= bineq, name="ineq")


# In[73]:


# Solve the model
model.optimize()


# In[74]:


# Get the optimal solution
x[:, 4] = model.getAttr('x')
cost[:, 4] = np.matmul(np.transpose(c),x[:, 4])


# In[75]:


# Find the facility with minimal cost
cost = list(cost)
# added 1 to this line where  calculate minimal to represent facilities from 1 to 5
minimal = cost.index(min(cost))+1 
print("This policy should be implemented at facility: "+ str(minimal))


# In[76]:


# facility 1 is Alexandria city


# In[ ]:




