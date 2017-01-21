import os,pandas,time
from dbutils import RunData,printlist

def data_files(num):
    files = os.listdir(RunData.datadir)
    paths = list()
    ignored = list()
    for f in files:
        p = os.path.join(RunData.datadir,f)
        try:
            paths.append((p,int(os.path.basename(f).strip('.pkl'))))
        except:
            ignored.append(p)
    if(len(ignored) > 0):
        print("ignored:")
        printlist(ignored)
    paths.sort(key=lambda x: x[1],reverse=True)
    if num > len(paths):
        num = len(paths) - 1
    return paths[:num]

def explore_rundata(num=5):
    paths = data_files(num)
    for i,p in enumerate(paths,1):
        print(str(i)+".",time.ctime(p[1]))
    choice = input("choose data to load: ")
    data = pandas.read_pickle(paths[int(choice) - 1][0])
    data.when()
    data.how()
    print("\nprint data.show() to view the results of this run")
    return data

def list_summary(num=10):
    paths = data_files(num)
    for p in paths:
        d = pandas.read_pickle(p[0])
        d.when()
        print(d)

if __name__ == '__main__':
    print("print data = explore_rundata(): to load a run as data")
    print("print list_summary(n) for a summary of n runs back")
