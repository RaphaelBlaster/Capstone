import numpy as np
from PIL import Image
from biomass_jepa.block_shuffle import shuffle_square_tile, permutation_seed

def test_block_shuffle():
    image = Image.fromarray(np.random.default_rng(12).integers(0,256,(1000,1000,3),dtype=np.uint8))
    for grid in (4,10):
        seed = permutation_seed('abc'*22, grid, 0, 0)
        a = shuffle_square_tile(image,grid,seed)
        b = shuffle_square_tile(image,grid,seed)
        assert a.tobytes() == b.tobytes()
        assert not np.array_equal(np.asarray(a),np.asarray(image))
        # Per-channel exact equality of histograms, verifying no loss/overlap.
        for ch in range(3):
            assert np.array_equal(np.bincount(np.asarray(a)[:,:,ch].ravel(), minlength=256),
                                  np.bincount(np.asarray(image)[:,:,ch].ravel(), minlength=256))
        assert len({permutation_seed('abc'*22,grid,k,v) for k in range(4) for v in range(2)}) == 8

if __name__=='__main__':
    test_block_shuffle()
    print('PASS: deterministic per-source 4x4/10x10 block permutation, pixel conservation')
