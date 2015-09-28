__author__ = 'Ivan'

from BeautifulSoup import BeautifulSoup

from pylatex import Document, Table

def parseRow(row):

    rowHead=row.th.text;

    measurements=list();
    measurements.append(rowHead)
    for r,idx in zip(row,range(0,len(row))):

        if idx>1:
            #print unicode(r.string)
            s=r.string
            if s!='\n':
                measurements.append(float(s))

    return (measurements)

def parseHTMLfileFromVOT2014(filename):
    f = open(filename, 'r')

    html=f.read();
    parsed_html = BeautifulSoup(html)

    table= parsed_html.table

    tableColumns=list()


    #print table

    data=list()
    idx=0
    for r in table:
        if r.string==None:

            if idx==1:
                for rr in r:
                    tableColumns.append(rr.string.rstrip())
            elif idx>1:

                parsedRow=parseRow(r)
                data.append(parsedRow)
            idx=idx+1



        #     print row
        #     continue;
        # elif idx>=3:
        #     # process header
        #     print row
        #     for r in row:
        #         print r
        #         tableColumns.append(r.text)
        #
        #
        #     break
        # else:
        #     parseRow(row)


    return (tableColumns,data)


def createSimpleTable(tableColumns,d,filename):
    doc = Document()

    with doc.create(Table('|c|c|c|c|c|c|c|c|c|')) as table:

        for row in d:

            row[0]=row[0].replace('_','\_')
            row[0]="\\textbf{"+row[0]+"}"
            table.add_hline()
            table.add_row(tuple(row))
        table.add_hline()

    a = open(filename, 'wr')
    table.dump(a)


def deleteColumns(l,from_idx,to_idx):

    r=list()

    for line in l:
        line_c=line
        if len(line_c)>to_idx:
            del line_c[to_idx+1:]

        del line_c[1:from_idx]



def sortAndColor(l):

    # l is a list


    indices=[i[0] for i in sorted(enumerate(l), key=lambda x: x[1])]

    # find indices
    s=[l[i] for i in indices]

    index=0

    c=s[index]

    resultList=l;
    for el,idx in zip(l,range(0,len(l))):
        if el==c:
            # replace el with red color
            el='\\textcolor{red}{'+str(el)+'}'
            resultList[idx]=el

    # go to next
    for el,idx in zip(s,range(0,len(l))):
        if el>c:
            c=s[idx]
            index=idx
            break

    for el, idx in zip(l, range(0, len(l))):
        if el == c:
            # replace el with red color
            el = '\\textcolor{blue}{' + str(el) + '}'
            resultList[idx] = el

    # go to next
    for el, idx in zip(s, range(0, len(l))):
        if el > c:
            c = s[idx]
            index = idx
            break

    for el, idx in zip(l, range(0, len(l))):
        if el == c:
            # replace el with red color
            el = '\\textcolor{green}{' + str(el) + '}'
            resultList[idx] = el

    return resultList

def colorTopResults(d):

    names=[x[0]for x in d]

    # d1=[x[1] for x in d];
    # d2 = [x[2] for x in d];
    # d3 = [x[3] for x in d];
    #
    # d1=sortAndColor(d1)
    # d2= sortAndColor(d2)
    # d3= sortAndColor(d3)


    d_colored=list()

    for n in names:
        l=list()
        l.append(n)
        d_colored.append(l)

    for idx in range(1,len(d[0])):

        d1=[x[idx] for x in d]
        d1 = sortAndColor(d1)
        for index in range(0,len(d1)):
            d_colored[index].append(d1[index])

    #d_colored=[(n,x,y,z) for n,x,y,z in zip(names,d1,d2,d3)]

    return d_colored


def createFullTable(tableColumns,d_challange,d_pool,d_weighted,filename):
    doc = Document()

    order=[x[0] for x in d_challange];

    # delete unnecessary data

    deleteColumns(d_challange,4,6)
    deleteColumns(d_pool,1,6)
    #deleteColumns(d_weighted,5,7)


    d_challange=colorTopResults(d_challange)
    d_pool= colorTopResults(d_pool)
    #d_weighted=colorTopResults(d_weighted)

    print d_challange
    print d_pool


    # find top 3 in each column replace 1 with red color, 2 with blue, 3 with red. If there is tie same color
    # should be used



    addBoldLate=lambda x: "\\textbf{" + x + "}"

    with doc.create(Table('|c|c|c|c|c|c|c|c|c|c|')) as table:

        topLine = list()

        table.add_hline()
        topLine.append("Experiment ")
        topLine.append("\multicolumn{3}{ | c |}{Baseline} ")
        topLine.append("\multicolumn{3}{ | c |}{Region pertubation} ")
        topLine.append("\multicolumn{3}{ | c |}{Averaged} ")

        table.add_row(topLine)

        normalizationLine=list()

        table.add_hline()
        normalizationLine.append("Normalization ")
        normalizationLine.append("\multicolumn{3}{ | c |}{sequence - pooled} ")
        normalizationLine.append("\multicolumn{3}{ | c |}{sequence - pooled} ")
        normalizationLine.append("\multicolumn{3}{ | c |}{per - attribute} ")

        table.add_row(normalizationLine)

        # \hline
        # Normalization & \multicolumn
        # {3}
        # { | c |}{sequence - pooled} & \multicolumn
        # {3}
        # { | c |}{sequence - pooled} & \multicolumn
        # {3}
        # { | c |}{per - attribute} \ \
        #         \hline
        table.add_hline()
        header3=list()

        header3.append('Ranking measure')
        for i in range(0,3):
            header3.append('A')
            header3.append('R')
            header3.append('Avg.')

        table.add_row(header3)
        table.add_hline()
        table.add_hline()

        for tracker in order:



            line1=[x for x in d_challange if x[0]==tracker][0]
            line2 = [x for x in d_pool if x[0] == tracker][0]


            tracker = tracker.replace('_', '\_')
            trackerName = tracker
            #table.add_hline()


            row=list()

            row.append(trackerName)

            row=row+line2[1:] + line1[1:]

            # find line corresponding
            table.add_row(row)
        table.add_hline()

    #print table
    a = open(filename, 'wr')
    table.dump(a)





if __name__ == "__main__":
    challange = '../../matlab/vot-toolkit/reports/report_vot2014_challenge/challenge.html'
    sequence_pool = '../../matlab/vot-toolkit/reports/report_vot2014_sequence_pool/article.html'
    sequence_weighted = '../../matlab/vot-toolkit/reports/report_vot2014_sequence_weighted/article.html'




    (tableColumns_challange,d_challange)=parseHTMLfileFromVOT2014(challange)
    (tableColumns_pool,d_pool) = parseHTMLfileFromVOT2014(sequence_pool)
    (tableColumns_weighted,d_weighted)= parseHTMLfileFromVOT2014(sequence_weighted)

    filename='fullTable.tex'
    folder='/Users/Ivan/Documents/Papers/My_papers/Tracking_with_Robust_Kalman/'
    #createSimpleTable(tableColumns,d,filename)

    createFullTable(tableColumns_challange,d_challange, d_pool,d_weighted, folder+filename)
    createFullTable(tableColumns_challange, d_challange, d_pool, d_weighted, filename)





