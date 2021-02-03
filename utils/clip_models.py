import torch
if __name__ == '__main__':
    model_path = './learn_rand.tar'
    a = torch.load(model_path,map_location='cpu')
    compatible_state_dict = {}
    for k,v in a['state_dict'].items():
        compatible_state_dict[k[7:]] = v
    torch.save(compatible_state_dict,model_path+'.clip.pth')
    