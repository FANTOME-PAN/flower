from flwr.client.abc_sa_client_wrapper import SAClientWrapper
from flwr.common.typing import SAServerMessageCarrier, SAClientMessageCarrier
from flwr.common.timer import Timer
import flwr.common.sec_agg_plus.sec_agg_client_logic as cl
import numpy as np
from typing import Dict
from flwr.common.typing import AskKeysRes, ShareKeysPacket, SAMessage
from flwr.client.client import Client


class SecAggPlusWrapper(SAClientWrapper):

    def __init__(self, c: Client):
        super().__init__(c)
        self.tm = Timer()

    def sa_respond(self, ins: SAServerMessageCarrier) -> SAClientMessageCarrier:
        self.tm.tic('s' + ins.identifier)
        msg = SAMessage()
        if ins.identifier == '0':
            pk1, pk2 = cl.setup_param(self, ins.str2scalar)
            # ret_msg = SAClientMessageCarrier('0', bytes_list=[pk1, pk2])
            msg.pk1 = pk1.decode('ascii')
            msg.pk2 = pk2.decode('ascii')
            ret_msg = SAClientMessageCarrier('0', sa_msg=msg)
        elif ins.identifier == '1':
            # key of received dict is string (json lib will silently convert any key type to string)
            share_keys_res_list = cl.share_keys(self, ins.sa_msg.public_keys_dict)
            msg.packets = [(o.source, o.destination, o.ciphertext.decode('ascii')) for o in share_keys_res_list]
            ret_msg = SAClientMessageCarrier('0', sa_msg=msg)
        elif ins.identifier == '2':
            # src_lst = ins.numpy_ndarray_list[0]
            # des_lst = ins.numpy_ndarray_list[1]
            # txt_lst = ins.bytes_list
            packet_lst = [ShareKeysPacket(s, d, t.encode('ascii')) for s, d, t in ins.sa_msg.packets]
            res = cl.ask_vectors(self, packet_lst, ins.fit_ins)
            ret_msg = SAClientMessageCarrier('2', parameters=res)
        elif ins.identifier == '3':
            actives, dropouts = ins.sa_msg.actives, ins.sa_msg.dropouts
            share_dict = cl.unmask_vectors(self, actives, dropouts)
            ret_msg = SAClientMessageCarrier('3', str2scalar=dict([(str(k), v) for k, v in share_dict.items()]))
        else:
            raise Exception("Invalid identifier")
        self.tm.toc('s' + ins.identifier)
        if self.sec_id == 6:
            f = open("log.txt", "a")
            f.write(f"Client without communication stage {ins.identifier}:{self.tm.get('s' + ins.identifier)} \n")
            if ins.identifier == '3':
                times = self.tm.get_all()
                f.write(f"Client without communication total: {sum([times['s0'], times['s1'], times['s2'], times['s3']])} \n")
            f.close()

        return ret_msg
