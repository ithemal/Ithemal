from sumtypes import sumtype, constructor

@sumtype
class LossReportMessage(object):
    LossReport = constructor('rank', 'loss', 'n_items')
