import json
import numpy as np
import matplotlib.pyplot as plt
from simplification.cutil import simplify_coords
from drawing import Drawing
from joblib import Parallel, delayed


class CreativeItem:
    def __init__(self, all_strokes: list):
        self.all_strokes = all_strokes

    @classmethod
    def from_json_item(cls, item: dict):
        return cls(all_strokes=item['all_strokes'])

    def plot(self):
        for bodypart in self.all_strokes:
            for stroke in bodypart:
                for line in stroke:
                    plt.plot([line[0], line[2]], [line[1], line[3]], c='k')
        plt.show()


class LineStringList:
    def __init__(self, linestring_list):
        self.linestring_list = linestring_list

    @classmethod
    def fromCreativeItem(cls, itm: CreativeItem):
        linestring_list = list()
        for bodypart in itm.all_strokes:
            if len(bodypart)==0:
                continue
            for stroke in bodypart:
                if len(stroke)<2:
                    continue
                line = stroke[0]
                linestring = [[line[0], line[1]]]
                for line in stroke:
                    linestring.append([line[2], line[3]])
                linestring_list.append(linestring)
        return cls(linestring_list)

    def simplify(self):
        new_list = list()
        for linestring in self.linestring_list:
            if len(linestring) > 2:
                new_list.append(simplify_coords(linestring, epsilon=1.0))
        self.linestring_list = new_list

    def plot(self):
        for linestring in self.linestring_list:
            plt.plot([p[0] for p in linestring], [p[1] for p in linestring], c='b')
        plt.show()

    def to_zero(self):
        x_min = min([min([p[0] for p in linesting]) for linesting in self.linestring_list])
        y_min = min([min([p[1] for p in linesting]) for linesting in self.linestring_list])
        new_list = [[[p[0] - x_min, p[1] - y_min] for p in linesting]for linesting in self.linestring_list]
        self.linestring_list = new_list

    def to_bbox(self, bmax=255):
        x_max = max([max([p[0] for p in linesting]) for linesting in self.linestring_list])
        y_max = max([max([p[1] for p in linesting]) for linesting in self.linestring_list])
        maxdim = max(x_max, y_max)
        coeff = bmax / maxdim
        new_list = [[[p[0]*coeff, p[1]*coeff] for p in linesting] for linesting in self.linestring_list]
        self.linestring_list = new_list

    def normalize(self):
        self.to_zero()
        self.to_bbox()


class DatasetItem:
    def __init__(self, emb):
        self.embedding = emb

    @classmethod
    def fromLineStringList(cls, lsl: LineStringList):
        pen_sequence = list()
        pen_position = [0, 0]
        for linestring in lsl.linestring_list:
            for point in linestring:
                pen_sequence.append([point[0] - pen_position[0], point[1] - pen_position[1], 0])
                pen_position = point
            pen_sequence[-1][2] = 1
        return cls(np.array(pen_sequence[1:], dtype=np.int16))

    def plot(self):
        firstpoint = self.embedding[0]
        plt.plot([0,firstpoint[0]], [0, firstpoint[1]], c='r')
        position = [0, 0]
        for i_point in range(len(self.embedding)-1):
            point = self.embedding[i_point]
            position = [position[0] + point[0], position[1] + point[1]]
            if point[2]==0:
                nextpoint = self.embedding[i_point + 1]
                nextposition = [position[0] + nextpoint[0], position[1] + nextpoint[1]]
                plt.plot([position[0], nextposition[0]], [position[1], nextposition[1]], c='r')
        plt.show()


def process_item(item):
    cc = CreativeItem.from_json_item(item)
    ll = LineStringList.fromCreativeItem(cc)
    ll.simplify()
    ll.normalize()
    dd = DatasetItem.fromLineStringList(ll)
    return dd.embedding


def process_creativebirds_json(
        in_path='raw_data/test.json',
        out_path="raw_data/test.npz",
):
    print("-- LOADING --")
    data = json.loads(open(in_path).read())
    resultlist = list()
    print(" -- PROCESSING --")
    resultlist = Parallel(n_jobs=4, verbose=2)(delayed(process_item)(item) for item in data)
    print("-- FINISHED --")
    n_items = len(resultlist)
    val_start = int(n_items*0.85)
    test_start = int(n_items*0.95)
    print(f"We have {n_items} items, splitting to train [:{val_start}], val[{val_start}:{test_start}], valid[{test_start}:]")
    np_result = np.array(resultlist, dtype=object)
    np.savez_compressed(out_path,
                        train=np_result[:val_start],
                        valid=np_result[val_start:test_start],
                        test=np_result[test_start:])


def test_one(idx=0, in_path='raw_data/test.json'):
    data = json.loads(open(in_path).read())
    # print(data)
    cc = CreativeItem.from_json_item(data[idx])
    cc.plot()
    ll = LineStringList.fromCreativeItem(cc)
    ll.simplify()
    ll.normalize()
    ll.plot()
    print(ll.linestring_list[:5])
    dd = DatasetItem.fromLineStringList(ll)
    print(dd.embedding[:5])
    dd.plot()
    print(dd.embedding.shape)
    print(dd.embedding.dtype)
    drawing = Drawing.from_npz_data(dd.embedding)
    drawing.plot()


def test_processing():
    inp = 'raw_data/test.json'
    outp = "raw_data/test.npz"
    process_creativebirds_json(in_path=inp, out_path=outp)
    a = np.load(outp, allow_pickle=True, encoding='latin1')
    drawing = Drawing.from_npz_data(a['train'][1])
    drawing.plot()


def main():
    process_creativebirds_json(in_path="raw_data/creative_birds_json.txt",
                               out_path="data/creativebirds.npz")


if __name__=="__main__":
    # test_processing()
    main()

    # a = np.load("data/creativebirds.npz", encoding='latin1', allow_pickle=True)
    # print(a['test'].shape)
    # print(a['test'][0].shape)
    # drawing = Drawing.from_npz_data(a['test'][0])
    # drawing.plot()
