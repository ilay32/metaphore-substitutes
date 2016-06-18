import os,pandas,time
from dbutils import RunData,printlist
def explore_rundata():
    files = os.listdir(RunData.datadir)
    paths = list()
    ignored = list()
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
            ignored.append(p)
    if(len(ignored) > 0):
        print("ignored:")
        printlist(ignored)
    choice = input("choose data to load: ")
    data = pandas.read_pickle(paths[int(choice) - 1])
    data.when()
    data.how()
    print("\nprint data.show() to view the results of this run")
    print("print x = explore_rundata(): to load another run as x")
    return data

if __name__ == '__main__':
    data = explore_rundata()
