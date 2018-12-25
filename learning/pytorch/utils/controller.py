from sumtypes import sumtype, constructor

@sumtype
class ControllerReport(object):
    ReportLoss = constructor('rank', 'loss', 'n_items')

@sumtype
class ControllerResponse(object):
    Ack = constructor()
    UpdateLr = constructor('lr')
