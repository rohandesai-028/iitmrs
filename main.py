import numpy as np
import matplotlib.pyplot as plt

print("\n\nHello, I'm a scheduler!!! :)\n\n")
# -----------------------------------------------------------------------
# Variable Declarations
# -----------------------------------------------------------------------
TOTAL_SLOTS = 10        #No of slots to schedule
TOTAL_MINI_SLOTS = 7     #No of minislots in one slot
TOTAL_PRBS  = 52        #Total available Physical Resource Blocks in one slot
TOTAL_EMBB_USERS = 10   #Total No of EMBB users to schedule
URLLC_RATE = 10
EMBB_ID = np.arange(0,TOTAL_EMBB_USERS,dtype=np.int32) #ID of EMBB users
CQI_TABLE = np.array([2*(120/1024), 
                          2*(157/1024), 
                          2*(193/1024), 
                          2*(251/1024),
                          2*(308/1024),
                          2*(379/1024),
                          2*(449/1024),
                          2*(526/1024),
                          2*(602/1024),
                          2*(679/1024),
                          4*(340/1024),
                          4*(378/1024),
                          4*(434/1024),
                          4*(490/1024),
                          4*(553/1024),
                          4*(616/1024),
                          4*(658/1024),
                          6*(438/1024),
                          6*(466/1024),
                          6*(517/1024),
                          6*(567/1024), 
                          6*(616/1024),
                          6*(666/1024),
                          6*(719/1024),
                          6*(772/1024),
                          6*(822/1024),
                          6*(873/1024),
                          6*(910/1024),
                          6*(948/1024)],dtype=np.float32)
OH = 0.14                       #Signalling overhead
MIMO_LAYERS = 4                 #No of MIMO Layers
NO_OF_SYM_PER_RB = 12
OFDM_SYM_DURATION = 1/14000     #OFDM Symbol Duration in seconds
OFDM_SYM_DURATION_MS = 1/14     #OFDM Symbol Duration in milliseconds

cqi_embb_user = np.around(np.random.normal(loc=18,scale=5,size=(TOTAL_EMBB_USERS)),decimals=0)
cqi_embb_user = np.clip(cqi_embb_user, 0, 28)
Rb_Map = np.zeros((TOTAL_SLOTS*TOTAL_MINI_SLOTS,TOTAL_PRBS,3),dtype=np.int32)
EMBB_buff = np.random.randint(low=70*60*100,high=70*100*100,size=TOTAL_EMBB_USERS) 
Bit_Rate = np.zeros(TOTAL_EMBB_USERS,dtype=np.float32)
Total_Data = np.zeros(TOTAL_EMBB_USERS,dtype=np.float32)
historical_br = np.ones(TOTAL_EMBB_USERS,dtype=np.float32)
served_curr_slot = np.zeros(TOTAL_EMBB_USERS,dtype=int)
# -----------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------
def PFScheduler(slot_num, prb, Rb_Map, EMBB_buff, Bit_Rate, Total_Data, historical_br):
    ''' 
    ------------------------------------------------------------------------------
    Description:  PF scheduler allocates PRB to different EMBB users based on 
                    their past bitrate information. 
    ------------------------------------------------------------------------------
    Input Arguments:
    RB Map    : Matrix of size=70,50,3
                RB Map is a numpy array of zeros. The scheduler allocates
                a PRB to a user, the RB Map is updated with the EMBB User ID.

    Bit Rate  : Array of size=EMBB_USERS
                This array stores the past bit rate of all the EMBB users.

    Tx Buffer : Array of size=EMBB_USERS
                This array contains the total data requested by the EMBB users.

    slot_num  : Current time slot of the allocation
    ------------------------------------------------------------------------------
    Output Arguments:
    RB Map    : Updated RB Map after resource block is allocated
    Bit Rate  : ''
    Tx Buffer : ''
    ------------------------------------------------------------------------------
    '''
    embb_user_to_sch = np.argmax(np.divide(Bit_Rate,historical_br))
    for slot_min in range(0,TOTAL_MINI_SLOTS):
        Rb_Map[(slot_num*7)+slot_min][prb][0] = embb_user_to_sch
    Total_Data[embb_user_to_sch] += Bit_Rate[embb_user_to_sch]
    historical_br[embb_user_to_sch] = Total_Data[embb_user_to_sch]/(slot_num+1)  
    EMBB_buff[embb_user_to_sch] -= Bit_Rate[embb_user_to_sch]
    served_curr_slot[embb_user_to_sch] = 1
#--------------------------------------------------------------------------------
def update_historical_br(served, br, TOTAL_EMBB_USERS, Total, slot_num):
    for embb_user in range(0,TOTAL_EMBB_USERS):
        if served[embb_user] == 0:
            br[embb_user] = Total[embb_user]/(slot_num+1)
    served = np.zeros(TOTAL_EMBB_USERS,dtype=int)
    return served, br    
#--------------------------------------------------------------------------------
def puncture_check(slot_num,):

    pass
#--------------------------------------------------------------------------------
def update_br():
    global Bit_Rate, TOTAL_EMBB_USERS, MIMO_LAYERS, NO_OF_SYM_PER_RB, CQI_TABLE, OFDM_SYM_DURATION_MS
    cqi_embb_user = np.around(np.random.normal(loc=18,scale=5,size=(TOTAL_EMBB_USERS)),decimals=0)
    cqi_embb_user = np.clip(cqi_embb_user, 0, 28)
    for embbs in range(0,TOTAL_EMBB_USERS):
        Bit_Rate[embbs] = (1-OH)*MIMO_LAYERS*NO_OF_SYM_PER_RB*CQI_TABLE[int(cqi_embb_user[embbs])]/OFDM_SYM_DURATION_MS

    pass
#--------------------------------------------------------------------------------
def urllc_arrival():
    embb_punct_ct = np.random.poisson(lam=URLLC_RATE)
    if embb_punct_ct < TOTAL_PRBS:
        punc = embb_punct_ct
    else:
        punc = TOTAL_PRBS
    total_urllc_dr = punc * (1-OH)*MIMO_LAYERS*NO_OF_SYM_PER_RB*CQI_TABLE[22]/OFDM_SYM_DURATION_MS/7

    pass
#--------------------------------------------------------------------------------
print("\nScheduler Start")
print("Number of EMBB Users", TOTAL_EMBB_USERS)
print("CQI of EMBB Users", cqi_embb_user)
print("EMBB Buffer data", EMBB_buff)
#print("RB map", Rb_Map)

for embbs in range(0,TOTAL_EMBB_USERS):
    Bit_Rate[embbs] = (1-OH)*MIMO_LAYERS*NO_OF_SYM_PER_RB*CQI_TABLE[int(cqi_embb_user[embbs])]/OFDM_SYM_DURATION_MS

print("Bit Rate of the EMBB users",Bit_Rate)
print("Historical Bit Rate",historical_br)
for slots in range(0,TOTAL_SLOTS):
    print("\n\nSlot Number",slots)
    for minislots in range(0,TOTAL_MINI_SLOTS):
        for prb in range(0,TOTAL_PRBS):
            if minislots == 0:
                PFScheduler(slots, prb, Rb_Map, EMBB_buff, Bit_Rate, Total_Data, historical_br)

            #print(Rb_Map)
        print("EMBB Users served:",served_curr_slot)
        served_curr_slot, historical_br = update_historical_br(served_curr_slot, historical_br, TOTAL_EMBB_USERS, Total_Data, slots)
        print("Total Data:",Total_Data)
        print("Req Buffer:",EMBB_buff)
        print("Historical BR:",historical_br)
