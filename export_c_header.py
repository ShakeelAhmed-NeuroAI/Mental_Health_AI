import torch
import numpy as np

# Load trained PyTorch checkpoint
checkpoint = torch.load("gat_eeg_model.pth", weights_only=False)
state = checkpoint['model_state_dict']

def fmt_array(tensor, var_name):
    arr = tensor.cpu().numpy().flatten()
    vals = ", ".join([f"{v:.6f}f" for v in arr])
    return f"const float {var_name}[{len(arr)}] = {{\n    {vals}\n}};\n"

with open("gat_weights.h", "w") as f:
    f.write("// Auto-generated GAT Model Weights for Bare-Metal Microcontroller C/C++ Deployment\n")
    f.write("#ifndef GAT_WEIGHTS_H\n#define GAT_WEIGHTS_H\n\n")
    f.write("#define GAT_IN_CHANNELS 5\n")
    f.write("#define GAT_HIDDEN_CHANNELS 16\n")
    f.write("#define GAT_HEADS 4\n")
    f.write("#define GAT_NUM_CLASSES 2\n\n")
    
    # Layer 1 Weights
    f.write(fmt_array(state['gat1.lin.weight'], "GAT1_LIN_WEIGHT"))
    f.write(fmt_array(state['gat1.att_src'], "GAT1_ATT_SRC"))
    f.write(fmt_array(state['gat1.att_dst'], "GAT1_ATT_DST"))
    f.write(fmt_array(state['gat1.bias'], "GAT1_BIAS"))
    f.write("\n")
    
    # Layer 2 Weights
    f.write(fmt_array(state['gat2.lin.weight'], "GAT2_LIN_WEIGHT"))
    f.write(fmt_array(state['gat2.att_src'], "GAT2_ATT_SRC"))
    f.write(fmt_array(state['gat2.att_dst'], "GAT2_ATT_DST"))
    f.write(fmt_array(state['gat2.bias'], "GAT2_BIAS"))
    f.write("\n")
    
    # Linear Classifier Weights
    f.write(fmt_array(state['classifier.weight'], "CLASSIFIER_WEIGHT"))
    f.write(fmt_array(state['classifier.bias'], "CLASSIFIER_BIAS"))
    
    f.write("\n#endif // GAT_WEIGHTS_H\n")

print("Successfully exported model weights to gat_weights.h!")
