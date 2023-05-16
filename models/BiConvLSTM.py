import torch
from torch import nn
from torch.autograd import Variable

class BiConvLSTMCell(nn.Module):
    """
    Generate a bidirectional convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, filter_size, bidirectional = False, useCuda = False, bias = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.padding = (filter_size-1)//2#in this way the output has the same size
        if(bidirectional == True):
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.useCuda = useCuda
        self.bias = bias 
        
        Gates = []
        for i in range(self.num_directions):
            conv = nn.Conv2d(self.input_size + self.hidden_size, 4 * self.hidden_size,
                               self.filter_size, padding=self.padding, bias = self.bias)
            if(self.useCuda):
                conv.cuda()
            Gates.append(conv)
            
        self.Gates = nn.ModuleList(Gates)
        
        
        
    def forward(self, input_, state = None):
        # get seq size
        seq_size = input_.data.size()[0]
        
        # generate empty prev_state, if None is provided
        if state is None:
            
            state = []
            if self.useCuda:
                for i in range(self.num_directions):
                    state.append(
                            (Variable(torch.zeros(self.state_size).cuda()),
                            Variable(torch.zeros(self.state_size).cuda()))
                        )
            else: 
                for i in range(self.num_directions):
                    state.append(
                        (Variable(torch.zeros(self.state_size)),
                        Variable(torch.zeros(self.state_size)))
                    )
        
        output = []
        new_state = []
        
        # data size is [sequence, batch, channel, height, width]
        for i in range(self.num_directions):
            
            prev_single_hidden, prev_single_cell = state[i]
            single_output = []
            gates = self.Gates[i]
                
            if i == 1:
                steps = range(seq_size - 1, -1, -1)
            else:
                steps = range(seq_size)
            for j in steps:
                stacked_inputs = torch.cat((input_[j], prev_single_hidden),1)
                gates_out = gates(stacked_inputs)
                # chunk across channel dimension
                in_gate, remember_gate, out_gate, cell_gate = gates_out.chunk(4, 1)
                # apply sigmoid non linearity
                in_gate = torch.sigmoid(in_gate)
                remember_gate = torch.sigmoid(remember_gate)
                out_gate = torch.sigmoid(out_gate) #torch.sigmoid al posto di f.sigmoid
                # apply tanh non linearity
                cell_gate = torch.tanh(cell_gate) #torch.tanh al posto di f.tanh
                # compute current cell and hidden state
                prev_single_cell = (remember_gate * prev_single_cell) + (in_gate * cell_gate)
                prev_single_hidden = out_gate * torch.tanh(prev_single_cell)
                single_output.append(prev_single_hidden.unsqueeze(0))
            
            single_output = torch.cat(single_output, 0)
            
            output.append(single_output)
            new_state.append((prev_single_hidden, prev_single_cell))
        
        output = torch.cat(output, 2)
        return output, new_state
        
    def initHidden(self, state_size):
        state = []
        if self.useCuda:
            for i in range(self.num_directions):
                state.append(
                        (Variable(torch.zeros(state_size).cuda()),
                        Variable(torch.zeros(state_size).cuda()))
                    )
        else: 
            for i in range(self.num_directions):
                state.append(
                    (Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size)))
                )
        return state
    
class BiConvLSTM(nn.Module):
    """
    Generate a bidirectional convolutional LSTM - hidden_size is the size of the hidden state of a single direction
    """

    def __init__(self, input_size, hidden_size, filter_size, num_layers, bidirectional = False, useCuda = False, bias = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.num_layers = num_layers
        if(bidirectional == True):
            self.num_directions = 2
        else:
            self.num_directions = 1
        cell_list = []
        cell_list.append(BiConvLSTMCell(self.input_size, self.hidden_size, self.filter_size, bidirectional, useCuda, bias)) #the first one has a different number of input channels
        
        for idcell in range(1,num_layers):
            cell_list.append(BiConvLSTMCell(self.hidden_size*self.num_directions, self.hidden_size, self.filter_size, bidirectional, useCuda, bias))
        
        self.cell_list=nn.ModuleList(cell_list) 
    
    def forward(self, input_, state = None):
        current_input = input_
        next_state = []
        
        if state == None:
            state = []#this is a list of tuples
            state_size = [input_.data.size()[1], self.hidden_size, input_.data.size()[3], input_.data.size()[4]]
            for i in range(self.num_layers):
                state.append(self.cell_list[i].initHidden(state_size))
            
        
        for idlayer in range(self.num_layers):#loop for every layer
            layer_state = state[idlayer]
            current_input, layer_next_state = self.cell_list[idlayer](current_input, layer_state)
            next_state.append(layer_next_state)
            
        return current_input,next_state
    