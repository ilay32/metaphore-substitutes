import os,pandas,time
from main import RunData
def explore_rundata():
    files = os.listdir(RunData.datadir)
    paths = list()
    listed = 0
    for f in files:
        p = os.path.join(RunData.datadir,f)
        try:
            intstamp = int(os.path.basename(f).strip('.pkl'))
            listed += 1
            t = time.ctime(intstamp)
            print(listed ,".",t)
            paths.append(p)
        except:
            print("skipping ",p)
    choice = input("choose data to load: ")
    data = pandas.read_pickle(paths[int(choice) - 1])
    data.when()
    data.how()
    print("print data.show() to view the results of this run")
    print("print explore_rundata(): to load another run")
    return data

if __name__ == '__main__':
    data = explore_rundata()
