
def progress_bar(now,length,notes='',frequency=1):
    '''
    now:current progress
    
    length:whole mission
    '''
    if int(now/(length)*100) % frequency == 0 :
        string = ' '*10+' '*len(notes)+'                \r'
        print(string,end='')
        print('{:*<10s} {:d}% {}\r'.format('>'*int(now/(length)*10),int(now/(length)*100),notes),end='')

