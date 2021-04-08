import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import PIL


class Drawing:
    """
    This class is for data visualization.
    """
    def __init__(self, emb_sequence):
        assert emb_sequence.shape[-1] == 5
        self.embedding_sequence = emb_sequence

    @classmethod
    def from_npz_data(cls, npzarray):
        newdata = np.zeros(shape=(npzarray.shape[0], 5), dtype=np.float32)
        newdata[:,:2] = npzarray[:,:2]
        newdata[:,2] = 1 - npzarray[:,2]
        newdata[:,3] = npzarray[:,2]
        newdata[-1,3] = 0
        newdata[-1,4] = 1
        return cls(newdata)

    @classmethod
    def from_tensor_prediction(cls, prediction):
        """
        :param prediction: (seq_len, batch_size=1, 5)
        :return:
        """
        return cls(prediction.squeeze(1).detach().cpu().numpy())

    def get_lines(self):
        current_position = np.array([0, 0], dtype=np.float32)
        lines_list = list()
        lines_stroke_id = list()
        stroke_id = 0
        # lines_list.append([current_position.tolist(), self.embedding_sequence[0, :2].tolist()])
        for i_point in range(len(self.embedding_sequence)-1):
            point = self.embedding_sequence[i_point]
            current_position += point[:2]
            if point[2]==1:
                nextpoint_position = current_position + self.embedding_sequence[i_point+1, :2]
                lines_list.append([current_position.tolist(), nextpoint_position.tolist()])
                lines_stroke_id.append(stroke_id)
            else:
                stroke_id += 1
        return lines_list, lines_stroke_id

    def render_image(self, show=False):
        lines, lines_id = self.get_lines()
        colors = ['k']
        if len(lines)>1:
            evenly_spaced_interval = np.linspace(0, 1, lines_id[-1]+1)
            colors = [mpl.cm.tab10(x) for x in evenly_spaced_interval]
        plt.axis('equal')
        plt.axis("off")
        for i in range(len(lines)):
            line = lines[i]
            plt.plot([line[0][0], line[1][0]], [-line[0][1], -line[1][1]], color=colors[lines_id[i]])
        if show:
            plt.show()

    def plot(self):
        self.render_image(show=True)

    def tensorboard_plot(self):
        self.render_image(show=False)
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                        canvas.tostring_rgb())
        plt.close("all")
        img_array = np.asarray(pil_image)
        return np.transpose(img_array, (2, 0, 1))

if __name__=="__main__":
    a = np.load("data/owl.npz", encoding='latin1', allow_pickle=True)
    # print(a['test'][0])
    drawing = Drawing.from_npz_data(a['valid'][0])
    print(a['test'][1].shape)
    # drawing = Drawing.from_npz_data(a['test'][1])
    plt.figure(figsize=(7,4))
    plt.suptitle("Epoch")
    plt.subplot(1,2,1)
    plt.title("orig")
    drawing.render_image(show=False)
    plt.subplot(1,2,2)
    plt.title("recon")
    drawing.render_image(show=False)
    plt.show()


    # image = drawing.tensorboard_plot()
    # plt.imshow(image.transpose(), cmap='gray_r')
    # plt.show()
    # print(drawing.get_lines())