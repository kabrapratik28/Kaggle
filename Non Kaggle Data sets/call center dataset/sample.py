from sets import Set
import matplotlib.pyplot as plt
#GUI support
from easygui import *

import pandas as pd 

#white not added as bckgrnd is white
colorsOfMatLib = ['b','g','r','c','m','y','k']
lenColorsOfMatLib = len(colorsOfMatLib)

#style of line
styleOfLineInMatLib = ['-','--',':','-.']
lenStyleOfLineInMatLib = len(styleOfLineInMatLib)

#marker of points
markerOfMatLib = ['+','.','o','*','p','s','x','D','h','^']
lenMarkerOfMatLib = len(markerOfMatLib)

def numberOfUniqueCustomers() : 
    #number of customers in the bank
    customers = Set()
    
    for eachMonth in months:
         customers.update(mapOfMonthData[eachMonth]["customer_id"])
    return len(customers)
    
'''    
#reference : http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot    
#by default parameter provided 
#    month 1 to 31
#    month name is optional 
#    plot lib => marker='o', linestyle='-', color='b'
'''
def plotCustomerAsPerDate(listOfNoOfCustomersOnThatDay,listOfdate=range(1,32),month='',marker='o', linestyle='-', color='b'): 
    plt.plot(listOfdate,listOfNoOfCustomersOnThatDay,marker=marker, linestyle=linestyle, color=color,label=month)

def showPlot(xlabel='',ylabel='',title=''):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)    
    #show right upper corner     
    #plt.legend()   #creating problem (when multiple selects) 
    #show bottom of figure    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
    plt.show()

def dateExtractor(date):
    year = date[0:2]
    month = date [2:4]
    day = date [4:6]
    return {'year':int(year),'month':int(month),'day':int(day)}    

def plotForGivenMonth(month,marker,linestyle,color):
    listOfCallsForJan = mapOfMonthData[month]["date"]
    #31 number of zeros
    noOfcallsEachDayCounter = [0]*31
    for eachDate in listOfCallsForJan:
        formatedDate = dateExtractor(eachDate)['day']
        #as index starts from zero (decrease date by 1)
        formatedDate = formatedDate -1
        #increase number of call by 1
        noOfcallsEachDayCounter[formatedDate] = noOfcallsEachDayCounter[formatedDate] +1
    
    plotCustomerAsPerDate(listOfNoOfCustomersOnThatDay=noOfcallsEachDayCounter,month=month,marker=marker, linestyle=linestyle, color=color)
    #print noOfcallsEachDayCounter
    #print sum(noOfcallsEachDayCounter)

def guiForSelectingMonths():
    msg ="What Months to Plot?"
    title = "Plotting of Month Data"
    choices = multchoicebox(msg, title,months)
    return choices

def noOfCallsEachDayPlotter():
    countermarker = 0
    counterLinestyle = 0 
    counterColor = 0 
    
    for eachMonth in guiForSelectingMonths():
        #print markerOfMatLib[countermarker], styleOfLineInMatLib[counterLinestyle],colorsOfMatLib[counterColor]
        plotForGivenMonth(eachMonth,marker=markerOfMatLib[countermarker],linestyle=styleOfLineInMatLib[counterLinestyle],color=colorsOfMatLib[counterColor])
        countermarker = (countermarker +1) % lenMarkerOfMatLib
        counterLinestyle = (counterLinestyle +1) % lenStyleOfLineInMatLib
        counterColor = (counterColor + 1) % lenColorsOfMatLib
        
    showPlot('Date','Number Of Customers call','Date vs Customer Call')
    

# *********** Get data ****************
maxcol = 17
    
#months to read
months = ["january","february","march","april","may","june","july","august","september","october","november","december"]
mapOfMonthData = {}

mapOfPandaMonthData = {}


for eachMonth in months:
    #open file
    dataset = open("Months\\"+eachMonth+".txt", "r")
        
    count = 0 
    
    mapofcolms = {}
    mapofindextocol = {}
    
    for line in dataset:
            
            line = line.strip("\n")
            #if not first line store the data
            #eachrow is list
            eachrow = line.split("\t")
            
            if count ==0 :
                #initilization of code.
                indexcounterofcol = 0 
                #print line #i.e.
                for clmname in eachrow:
                    mapofcolms[clmname] = []
                    mapofindextocol[indexcounterofcol] = clmname
                    indexcounterofcol = indexcounterofcol + 1
                #print mapofcolms
                #print mapofindextocol
            else:
                #validation that, list size is equal to maxcol
                listCount = 0 
                if len(eachrow) == maxcol :
                    #nice data
                    for eachElement in eachrow : 
                        clmName = mapofindextocol[listCount]
                        mapofcolms[clmName].append(eachElement)
                        listCount = listCount + 1
                else:
                    #bad data
                    #print "very bad data : "+ str(eachrow) +eachMonth 
                    None
                    
            count = count + 1
            #if count > 5 : 
            #        break
    mapOfMonthData[eachMonth] = mapofcolms            
    #print len(mapofcolms['outcome'])  #printing no of data rows each month has.
    mapOfPandaMonthData[eachMonth] = pd.DataFrame((mapofcolms))
#display graph
#noOfCallsEachDayPlotter()



'''
listOfDate = mapOfMonthData['january']['date'][:5]
listOfTime = mapOfMonthData['january']['vru_entry'][:5]

print listOfDate,listOfTime



listOfDateTime = []


for eachDateTimeIndex in range(0,len(listOfDate)):
    dateTime = listOfDate[eachDateTimeIndex] + ' '+listOfTime[eachDateTimeIndex]
    listOfDateTime.append(dateTime)
    


df = pd.DataFrame(({'month_date': pd.to_datetime(listOfDateTime), 'count':[1]*len(listOfDateTime) }))

df = df.set_index('month_date')

print df.index.hour


'''
def graphOfCallsPerHour(month):
    df = mapOfPandaMonthData[month]
    #date and time 
    df['dateTime'] = df['date'] + " " + df['vru_entry']
    #str to datetime
    df['dateTime']=pd.to_datetime(df['dateTime'])
    #index
    df = df.set_index('dateTime')
    #make clm with each entry as 1 
    df['count'] = 1
    
    #take date in another date frame
    df2 = pd.DataFrame({'hour':df.index.hour,'noofcalls':df['count']})
    
    df3 = df2.groupby('hour').noofcalls.sum()

    return df3

df = mapOfPandaMonthData['january']

#date and time 
df['dateTime'] = df['date'] + " " + df['vru_entry']
#str to datetime
df['dateTime']=pd.to_datetime(df['dateTime'])
#index
df = df.set_index('dateTime')
#make clm with each entry as 1 
df['count'] = 1

df2 = df.loc[df['server'] == 'MICHAL']

df2 = pd.DataFrame({'hour':df2.index.hour,'noofcalls':df2['count']})

df3 = df2.groupby('hour').noofcalls.sum()

df3.plot(kind='bar')

df3['xin'] = graphOfCallsPerHour('january')