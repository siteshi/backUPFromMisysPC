#!/usr/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
class Employee:
    empCount =0;
    def __init__(self,name,salary,sex):

        self.name =name
        self.salary = salary
        self.sex =sex
        self.empCount = self.empCount +1

    def displayCount(self):
        print "total employee: %d" % self.empCount
    def displayDetails(self):
        print "name: %s salary: %d and sex: %s " %(self.name,self.salary,self.sex)


if __name__ == '__main__':
    emp1 = Employee("Zara1", 1000,'Male')
    emp2 = Employee("Zara2", 2000,'Female')
    emp3 = Employee("Zara3", 3000,'NA')
    emp1.displayCount()
    emp2.displayCount()
    emp3.displayCount()
    print "Employee.__doc__:", Employee.__doc__
    print "Employee.__name__:", Employee.__name__
    print "Employee.__module__:", Employee.__module__
    print "Employee.__bases__:", Employee.__bases__
    print "Employee.__dict__:", Employee.__dict__

    ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
    ts = ts.cumsum()
    ts.plot(label="Predicted GPU executionTime")
    plt.show()