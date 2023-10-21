# Written by NanobotZ
# Modified by MediaMoots

from io import BufferedReader, BufferedWriter, IOBase, BytesIO
import struct
import copy
import binascii
from typing import List, Tuple, Union

def calculate_crc32_hash(input_string):
    return binascii.crc32(input_string.encode('utf8'))

def pad_count(pos: int, multiplier: int = 64) -> int:
    assert pos >= 0
    diff = (pos % multiplier)
    if diff == 0:
        return 0
    return multiplier - diff

def pad_till(pos: int, multiplier: int = 64) -> int:
    assert pos >= 0
    return pos + pad_count(pos, multiplier)

def pad_to_file(writer: BufferedWriter, multiplier: int = 64) -> None:
    diff = pad_count(writer.tell(), multiplier)
    if diff != 0 and diff != multiplier: # append only if needed
        writer.write(b'\x00' * diff)

def get_file_size(io_object: IOBase) -> int:
    pos = io_object.tell()
    io_object.seek(0, 2) # seek to the end of the file
    end = io_object.tell()
    io_object.seek(pos, 0) # seek back
    return end

def get_high_nibble(byte: int) -> int:
    return byte >> 4 & 0x0F

def get_low_nibble(byte: int) -> int:
    return byte & 0x0F

def pad_to_4_byte_boundary(data):
    # Calculate the number of bytes needed to reach a 4-byte boundary
    padding_length = (4 - (len(data) % 4)) % 4
    
    # Append the padding bytes
    padding = b'\x00' * padding_length
    
    # Combine the data and padding
    padded_data = data + padding
    
    return padded_data

class AmtaUnknownSection:
    def __init__(self, reader: BufferedReader, bom: str) -> None:       
        self.unk_1: int
        self.unk_2: float
        self.unk_3: float
        self.unk_4: float
        self.unk_5: float
        self.unk_6: float
        
        if reader == None:
            return

        self.unk_1, self.unk_2, self.unk_3, self.unk_4, self.unk_5, self.unk_6 = struct.unpack(bom + "I5f", reader.read(24))

    def write(self, writer: BufferedWriter, bom: str) -> None:
        writer.write(struct.pack(bom + "I5f", self.unk_1, self.unk_2, self.unk_3, self.unk_4, self.unk_5, self.unk_6))
        
    def to_bytes(self, bom: str):
        return struct.pack(bom + "I5f", self.unk_1, self.unk_2, self.unk_3, self.unk_4, self.unk_5, self.unk_6)

    def get_size(self) -> int:
        return 24 # content length doesn't change
    
class AmtaUnknown2Record:
    def __init__(self, reader: BufferedReader, bom: str) -> None:       
        self.unk_1: int
        self.unk_2: int
        self.unk_3: int
        self.unk_4: int
        
        if reader == None:
            return

        self.unk_1, self.unk_2, self.unk_3, self.unk_4 = struct.unpack(bom + "4I", reader.read(16))

    def write(self, writer: BufferedWriter, bom: str) -> None:
        writer.write(struct.pack(bom + "4I", self.unk_1, self.unk_2, self.unk_3, self.unk_4))
        

    def get_size(self) -> int:
        return 16 # content length doesn't change
    
class AmtaUnknown2Section:
    def __init__(self, reader: BufferedReader, bom: str) -> None:
        self.count: int
        self.records: List[AmtaUnknown2Record] = []

        self.count, = struct.unpack(bom + "I", reader.read(4))
        for _ in range(self.count):
            self.records.append(AmtaUnknown2Record(reader, bom))

    def write(self, writer: BufferedWriter, bom: str) -> None:
        writer.write(struct.pack(bom + "I", self.count))
        for record in self.records:
            record.write(writer, bom)

    def get_size(self) -> int:
        return 4 + self.count * 16

class Amta:
    def __init__(self, reader: BufferedReader) -> None:       
        self.magic: bytes
        self.bom: str
        self.version_minor: int
        self.version_major: int
        self.size: int
        self.empty_offset: int # always 0
        self.UNKNOWN_offset: int
        self.UNKNOWN2_offset: int
        self.MINF_offset: int
        self.STRINGS_offset: int
        self.empty_offset_2: int # always 0
        self.DATA_size: int
        self.name_crc: int
        self.flags: int # TODO this is wrong
        self.tracks_per_channel: int # TODO this is wrong
        self.channel_count: int # TODO this is wrong
        self.rest_of_data: bytes # TODO find out what this holds, variable size
        self.UNKNOWN_section: AmtaUnknownSection
        self.UNKNOWN2_section: AmtaUnknown2Section = None
        self.rest_of_file: bytes # TODO find out what this holds, usually only the name, sometimes with some other string(s) separated by null byte
        self.name: str # don't write this to file

        if reader == None:
            return

        start_pos = reader.tell()

        self.magic = reader.read(4)
        assert self.magic == b'AMTA'

        bom = reader.read(2)
        assert bom == b'\xFE\xFF' or bom == b'\xFF\xFE'
        self.bom = '>' if bom == b'\xFE\xFF' else '<'

        self.version_minor, self.version_major, self.size = struct.unpack(self.bom + '2BI', reader.read(6))
        self.empty_offset, self.UNKNOWN_offset, self.UNKNOWN2_offset, self.MINF_offset, self.STRINGS_offset, self.empty_offset_2 = struct.unpack(self.bom + '6I', reader.read(24))

        # TODO implement reading (AND WRITING) like the rest of AMTA, this code is dirty ;d
        off_pos = reader.tell()
        reader.seek(start_pos + self.UNKNOWN_offset)
        self.UNKNOWN_section = AmtaUnknownSection(reader, self.bom)
        if self.UNKNOWN2_offset > 0:
            reader.seek(start_pos + self.UNKNOWN2_offset)
            self.UNKNOWN2_section = AmtaUnknown2Section(reader, self.bom)
        reader.seek(off_pos)

        # start of DATA section
        self.DATA_size, self.name_crc, self.flags, self.tracks_per_channel, self.channel_count = struct.unpack(self.bom + '3I2B', reader.read(14))
        self.rest_of_data = reader.read(self.DATA_size - 14) # 14 bytes read earlier

        self.rest_of_file = reader.read(self.size - (self.DATA_size + 0x24)) # read till end of AMTA section
        self.name = self.rest_of_file.split(b'\x00')[0].decode("ASCII")

    def write(self, writer: BufferedWriter) -> None:
        writer.write(self.magic) # 4
        writer.write(b'\xFE\xFF' if self.bom == '>' else b'\xFF\xFE') # 2
        writer.write(struct.pack(self.bom + '2BI', self.version_minor, self.version_major, self.size)) # 6
        writer.write(struct.pack(self.bom + '6I', self.empty_offset, self.UNKNOWN_offset, self.UNKNOWN2_offset, self.MINF_offset, self.STRINGS_offset, self.empty_offset_2)) # 24
        writer.write(struct.pack(self.bom + '3I2B', self.DATA_size, self.name_crc, self.flags, self.tracks_per_channel, self.channel_count)) # 14
        writer.write(self.rest_of_data)
        writer.write(self.rest_of_file)

    def get_size(self) -> int:
        return 4 + 2 + 6 + 24 + 14 + len(self.rest_of_data) + len(self.rest_of_file) # one element per write
        # return self.size # content length doesn't change YET

class BwavFileHeader: #https://gota7.github.io/Citric-Composer/specs/binaryWav.html
    def __init__(self, reader: BufferedReader) -> None:
        self.magic: bytes
        self.bom: str
        self.version_minor: int
        self.version_major: int
        self.crc: int
        self.is_prefetch: bool
        self.num_channels: int


        self.magic = reader.read(4)
        assert self.magic == b'BWAV'

        bom = reader.read(2)
        assert bom == b'\xFE\xFF' or bom == b'\xFF\xFE'
        self.bom = '>' if bom == b'\xFE\xFF' else '<'

        self.version_minor, self.version_major, self.crc, prefetch, self.num_channels = struct.unpack(self.bom + 'BBIHH', reader.read(10))
        self.is_prefetch = prefetch == 1

        assert self.num_channels > 0

    def write(self, writer: BufferedWriter) -> None:
        writer.write(self.magic) # 4
        writer.write(b'\xFE\xFF' if self.bom == '>' else b'\xFF\xFE') # 2
        writer.write(struct.pack(self.bom + 'BBIHH', self.version_minor, self.version_major, self.crc, 1 if self.is_prefetch else 0, self.num_channels)) # 10

    def get_size(self) -> int:
        return 16 # content length doesn't change

class BwavChannelInfo: #https://gota7.github.io/Citric-Composer/specs/binaryWav.html
    def __init__(self, reader: BufferedReader, bom: str) -> None:
        self.codec: int
        self.channel_pan: int
        self.sample_rate: int
        self.num_samples_nonprefetch: int
        self.num_samples_this: int
        self.dsp_adpcm_coefficients: bytes
        self.absolute_start_samples_nonprefetch: int
        self.absolute_start_samples_this: int
        self.is_looping: bool
        self.loop_end_sample: int
        self.loop_start_sample: int
        self.predictor_scale: int #?
        self.history_sample_1: int #?
        self.history_sample_2: int #?
        self.padding: int

        self.codec, self.channel_pan, self.sample_rate, self.num_samples_nonprefetch, self.num_samples_this = struct.unpack(bom + '2H3I', reader.read(16))
        self.dsp_adpcm_coefficients = reader.read(32) # TODO read with BOM!!!
        self.absolute_start_samples_nonprefetch, self.absolute_start_samples_this, \
            is_looping, self.loop_end_sample, self.loop_start_sample, self.predictor_scale, \
            self.history_sample_1, self.history_sample_2, self.padding = struct.unpack(bom + '5I4H', reader.read(28))
        self.is_looping = is_looping == 1

    def write(self, writer: BufferedWriter, bom: str) -> None:
        writer.write(struct.pack(bom + '2H3I', self.codec, self.channel_pan, self.sample_rate, self.num_samples_nonprefetch, self.num_samples_this)) # 16
        writer.write(self.dsp_adpcm_coefficients) # TODO write with BOM!!!
        writer.write(struct.pack(bom + '5I4H', self.absolute_start_samples_nonprefetch, self.absolute_start_samples_this, \
            1 if self.is_looping else 0, self.loop_end_sample, self.loop_start_sample, self.predictor_scale, \
            self.history_sample_1, self.history_sample_2, self.padding)) # 28
        
    def get_size(self) -> int:
        return 76 # content length doesn't change

class Bwav: #https://gota7.github.io/Citric-Composer/specs/binaryWav.html
    def __init__(self, path_or_bufferedReader: Union[str, BufferedReader], size: int = None) -> None:
        """'size' must be passed if bufferedReader was passed in 'path_or_bufferedReader'"""
        self.header: BwavFileHeader
        self.channel_infos: List[BwavChannelInfo] = []
        self.channel_samples: List[bytes] = []


        reader: BufferedReader
        reader_opened_here = False
        if isinstance(path_or_bufferedReader, str):
            self.filepath = path_or_bufferedReader
            reader = open(path_or_bufferedReader, "rb")
            reader_opened_here = True
        else:
            reader = path_or_bufferedReader

        if not reader_opened_here and not size:
            raise ValueError("'size' must be passed if bufferedReader was passed in 'path_or_bufferedReader'")
        
        if reader_opened_here and not size:
            size = get_file_size(reader)
        
        pos = reader.tell()
        self.header = BwavFileHeader(reader)
        for _ in range(self.header.num_channels):
            self.channel_infos.append(BwavChannelInfo(reader, self.header.bom))

        samples_per_channel_to_read = size - self.channel_infos[-1].absolute_start_samples_this
        
        for channel in self.channel_infos:
            reader.seek(pos + channel.absolute_start_samples_this)
            self.channel_samples.append(reader.read(samples_per_channel_to_read) if samples_per_channel_to_read > 0 else b'')

        if reader_opened_here:
            reader.close()

        self.decoded_channels: List[List[int]] = [None] * self.header.num_channels

    def write(self, path_or_bufferedWriter: Union[str, BufferedWriter]):
        writer: BufferedWriter = None
        writer_opened_here = False

        if isinstance(path_or_bufferedWriter, str):
            writer = open(path_or_bufferedWriter, "wb")
            writer_opened_here = True
        else:
            writer = path_or_bufferedWriter

        pos = writer.tell()
        self.header.write(writer)
        for channel in self.channel_infos:
            channel.write(writer, self.header.bom)

        for idx, channel in enumerate(self.channel_infos):
            writer.seek(pos + channel.absolute_start_samples_this)
            writer.write(self.channel_samples[idx])

        if writer_opened_here:
            writer.close()

    def get_size(self) -> int:
        header_and_info_part = self.header.get_size() + sum([channel.get_size() for channel in self.channel_infos])

        # get only unique samples, as some channels can point to the same sample array
        unique_samples: List[Tuple[int, int]] = []
        for idx in range(self.header.num_channels):
            if not [idx_offset_tuple for idx_offset_tuple in unique_samples if idx_offset_tuple[1] == self.channel_infos[idx].absolute_start_samples_this]:
                unique_samples.append((idx, self.channel_infos[idx].absolute_start_samples_this))
        last_idx = unique_samples[-1][0]

        if len(unique_samples) != self.header.num_channels:
            pass

        samples_part = sum([pad_till(len(self.channel_samples[idx])) if idx != last_idx else len(self.channel_samples[idx]) for idx, _ in unique_samples])
        # condition in the line above - the last channel's samples don't need to be padded, but must remember about it if this BWAV is not the last one in BARS - caller must worry about it
        return pad_till(header_and_info_part) + samples_part if samples_part > 0 else header_and_info_part
            

    def convert_to_prefetch(self) -> bool:
        if self.header.is_prefetch:
            return True
        
        codec_samples = [0x1000, 0x3800, 0x9000]
        codec_bytes = [0x2000, 0x2000, 0x12200]

        converted = False
        for idx, channel in enumerate(self.channel_infos):
            req_samples = codec_samples[channel.codec]
            req_bytes = codec_bytes[channel.codec]

            if channel.num_samples_this < req_samples and len(self.channel_samples[idx]) < req_bytes:
                continue
            
            channel.num_samples_this = req_samples
            self.channel_samples[idx] = self.channel_samples[idx][:req_bytes]
            channel.absolute_start_samples_this = self.channel_infos[0].absolute_start_samples_this + (idx * req_bytes)
            converted = True

        if converted:
            self.header.is_prefetch = True

        return converted
    
    def decode_channel(self, channel_idx: int) -> List[int]:
        """returns a list of PCM16 (short) samples"""
        assert channel_idx < self.header.num_channels

        if self.decoded_channels[channel_idx]:
            return self.decoded_channels[channel_idx]

        src = self.channel_samples[channel_idx]
        dst: List[int] = []

        channel_info = self.channel_infos[channel_idx]
        if channel_info.codec == 0:
            for i in range(channel_info.num_samples_this):
                dst.append(*struct.unpack(self.header.bom + "h", src[i*2:i*2+2]))
        elif channel_info.codec == 1: # based on https://github.com/Thealexbarney/DspTool/blob/master/dsptool/decode.c
            samples = channel_info.num_samples_this
            hist1 = channel_info.history_sample_1
            hist2 = channel_info.history_sample_2
            coefs = struct.unpack(self.header.bom + '16h', channel_info.dsp_adpcm_coefficients)

            SAMPLES_PER_FRAME = 14
            frame_count = (samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
            samples_remaining = samples

            idx_src = 0
            for _ in range(frame_count):
                predictor = get_high_nibble(src[idx_src])
                scale = 1 << get_low_nibble(src[idx_src])
                idx_src += 1
                coef1 = coefs[predictor * 2]
                coef2 = coefs[predictor * 2 + 1]

                samples_to_read = min(SAMPLES_PER_FRAME, samples_remaining)
                for s in range(samples_to_read):
                    sample = 0
                    if s % 2 == 0:
                        sample = get_high_nibble(src[idx_src])
                    else:
                        sample = get_low_nibble(src[idx_src])
                        idx_src += 1
                    sample = sample - 16 if sample >= 8 else sample
                    sample = (((scale * sample) << 11) + 1024 + (coef1 * hist1 + coef2 * hist2)) >> 11

                    final_sample = sample
                    if final_sample > 32767: # short max val
                        final_sample = 32767
                    elif final_sample < -32768: # short min val
                        final_sample = -32768

                    hist2 = hist1
                    hist1 = final_sample
                    dst.append(final_sample)
                
                samples_remaining -= samples_to_read
        elif channel_info.codec == 2:
           raise NotImplementedError("Decoding Opus not implemented yet")

        self.decoded_channels[channel_idx] = dst
        return dst
    
    def decode(self) -> List[List[int]]:
        """returns a list of lists of PCM16 (short) samples, one list per channel"""
        result: List[List[int]] = []
        for channel_idx in range(self.header.num_channels):
            result.append(self.decode_channel(channel_idx))

        return result
    
    def export_wave(self, path: str) -> None:
        import wave
        with wave.open(path, "wb") as output:
            samples = self.decode()

            # wanted to use zip(), but it was so hecking slow, this is like 100* faster
            pack_format = self.header.bom + "h"
            data = bytearray()
            for i in range(self.channel_infos[0].num_samples_this):
                for chan in range(self.header.num_channels):
                    data.extend(struct.pack(pack_format, samples[chan][i]))

            output.setnchannels(self.header.num_channels)
            output.setframerate(self.channel_infos[0].sample_rate)
            output.setsampwidth(2)
            output.writeframes(data)

    
    def print_info(self) -> None:
        channel_pan_names = ["Left", "Right", "Middle", "Sub", "Side left", "Side right", "Rear ledt", "Rear right"]

        print(f'Magic:       {self.header.magic}')
        print(f'BOM:         {"Little Endian" if self.header.bom == "<" else "Big Endian"}')
        print(f'Version:     {self.header.version_major}.{self.header.version_minor}')
        print(f'CRC:         {self.header.crc}')
        print(f'Is prefetch: {self.header.is_prefetch}')
        print(f'Channels:    {self.header.num_channels}')

        for idx, channel in enumerate(self.channel_infos):
            print(f'\tChannel:                    {idx}')
            print(f'\tCodec:                      {channel.codec}')
            print(f'\tPan:                        {channel_pan_names[channel.channel_pan]}')
            print(f'\tSample rate:                {channel.sample_rate}')
            print(f'\tSamples non-prefetch:       {channel.num_samples_nonprefetch}')
            print(f'\tSamples this:               {channel.num_samples_this}')
            #print(f'\tADPCM coefficients:         {channel.dsp_adpcm_coefficients}')
            print(f'\tSamples Start non-prefetch: {channel.absolute_start_samples_nonprefetch}')
            print(f'\tSamples Start this:         {channel.absolute_start_samples_this}')
            print(f'\tIs looping:                 {channel.is_looping}')
            print(f'\tLoop end sample:            {channel.loop_end_sample}')
            print(f'\tLoop start sample:          {channel.loop_start_sample}')
            #print(f'\tPredictor scale:            {channel.predictor_scale}') # who cares about these 3 anyway
            #print(f'\tHistory sample 1:           {channel.history_sample_1}')
            #print(f'\tHistory sample 2:           {channel.history_sample_2}')
            print()

class Bars:
    def __init__(self, path_or_bufferedReader: Union[str, BufferedReader]) -> None:
        self.magic: bytes
        self.size: int
        self.bom: str
        self.version_minor: int
        self.version_major: int
        self.asset_count: int
        self.crc_hashes: List[int] = []
        self.meta_offsets: List[int] = []
        self.asset_offsets: List[int] = []
        self.unknown: bytes # TODO no idea what it is, it can be different size between different BARS, even with the same amount of assets
        self.metas: List[Amta] = []
        self.assets: List[Bwav] = []
        self.filepath: str = None # don't write to file, only assigned when path was provided


        reader: BufferedReader
        reader_opened_here = False
        if isinstance(path_or_bufferedReader, str):
            self.filepath = path_or_bufferedReader
            reader = open(path_or_bufferedReader, "rb")
            reader_opened_here = True
        else:
            reader = path_or_bufferedReader
        
        self.magic = reader.read(4)
        assert self.magic == b'BARS'

        size = reader.read(4)

        bom = reader.read(2)
        assert bom == b'\xFE\xFF' or bom == b'\xFF\xFE'
        self.bom = '>' if bom == b'\xFE\xFF' else '<'

        version = reader.read(2)

        asset_count = reader.read(4)

        self.size, self.version_minor, self.version_major, self.asset_count = struct.unpack(self.bom + 'I2BI', size + version + asset_count)

        self.crc_hashes.extend(struct.unpack(self.bom + 'I' * self.asset_count, reader.read(4 * self.asset_count)))

        for _ in range(self.asset_count):
            meta_offset, asset_offset = struct.unpack(self.bom + '2I', reader.read(8))
            self.meta_offsets.append(meta_offset)
            self.asset_offsets.append(asset_offset)

        self.unknown = reader.read(self.meta_offsets[0] - reader.tell())

        for meta_offset in self.meta_offsets:
            reader.seek(meta_offset)
            amta = Amta(reader)
            self.metas.append(amta)

        for asset_offset in self.asset_offsets:
            # # finding the read_size is annoying, because multiple indexes can point to the same asset ;d
            higher_offsets = [offset for offset in self.asset_offsets if offset > asset_offset]
            read_size = (min(higher_offsets) if higher_offsets else self.size) - asset_offset

            reader.seek(asset_offset)
            bwav = Bwav(reader, read_size)
            self.assets.append(bwav)

        if reader_opened_here:
            reader.close()

    def write(self, path_or_bufferedWriter: Union[str, BufferedWriter]):
        writer: BufferedWriter = None
        writer_opened_here = False

        if isinstance(path_or_bufferedWriter, str):
            writer = open(path_or_bufferedWriter, "wb")
            writer_opened_here = True
        else:
            writer = path_or_bufferedWriter

        writer.write(self.magic) # 4
        writer.write(struct.pack(self.bom + 'I', self.size)) # 4
        writer.write(b'\xFE\xFF' if self.bom == '>' else b'\xFF\xFE') # 2
        writer.write(struct.pack(self.bom + '2BI', self.version_minor, self.version_major, self.asset_count)) # 6

        writer.write(struct.pack(self.bom + 'I' * self.asset_count, *self.crc_hashes)) # 4 * self.asset_count

        for idx in range(self.asset_count):
            writer.write(struct.pack(self.bom + '2I', self.meta_offsets[idx], self.asset_offsets[idx])) # 8 * self.asset_count

        writer.write(self.unknown)

        for idx, meta_offset in enumerate(self.meta_offsets):
            writer.seek(meta_offset)
            self.metas[idx].write(writer)

        for idx, asset_offset in enumerate(self.asset_offsets):
            writer.seek(asset_offset)
            self.assets[idx].write(writer)

        if writer_opened_here:
            writer.close()

    def get_size(self) -> int:
        header_crc_metas_part = pad_till(4 + 4 + 2 + 6 + (4 * self.asset_count) + (8 * self.asset_count) + len(self.unknown) + sum([meta.get_size() for meta in self.metas]))

        # get only unique assets, as some metas can point to the same asset
        unique_assets: List[Tuple[int, int]] = []
        for idx in range(self.asset_count):
            if not [idx_offset_tuple for idx_offset_tuple in unique_assets if idx_offset_tuple[1] == self.asset_offsets[idx]]:
                unique_assets.append((idx, self.asset_offsets[idx]))
            
        assets_part = 0  
        if self.asset_count > 0:
            last_idx = unique_assets[-1][0]

            # condition below - the last BWAV doesn't need to be padded
            assets_part = sum([pad_till(self.assets[idx].get_size()) if idx != last_idx else self.assets[idx].get_size() for idx, _ in unique_assets])

        full_size = header_crc_metas_part + assets_part

        return full_size
    
    def replace_bwav(self, bwav_path: str, resize_if_needed: bool = False) -> bool:
        import pathlib
        name = pathlib.Path(bwav_path).stem

        found = False
        for idx, meta in enumerate(self.metas):
            if meta.name == name:
                found = True
                break
        
        if not found:
            print(f"Couldn't find '{name}' in this BARS file, skipping...")
            return False
        
        bwav_new = Bwav(bwav_path)
        bwav_old = self.assets[idx]

        if not resize_if_needed:
            if bwav_new.header.num_channels != bwav_old.header.num_channels:
                print(f"{name} - Replacing a BWAV with amount of channels different than original is disabled due to size differences")
                return False
        
            if  bwav_new.channel_infos[0].codec != bwav_old.channel_infos[0].codec:
                print(f"{name} - Replacing a BWAV with codec different than original is disabled due to size differences")
                return False
        
        if bwav_old.header.is_prefetch:
            # Perhaps we should allow the action below?
            # In theory, it will result in smaller file sizes, but might cause confusion due to the other audio in Resources/Streams not being needed (?)
            # Need to check what happens when prefetch gets replaced with non-prefetch
            
            # if not bwav_new.convert_to_prefetch():
            #     print(f"{name} - Couldn't convert the new BWAV to prefetch...")
            #     return False
            # else:
            print(f"{name} - Automatically converted BWAV to prefetch...")

        else:
            if bwav_new.header.is_prefetch:
                print(f"{name} - Can't replace a non-prefetch BWAV with a prefetch one!")
                return False
            
            if not resize_if_needed:
                print(f"{name} - Replacing a non-prefetch BWAV is disabled due to size differences")
                return False
            else:
                print(f"{name} - Replacing a non-prefetch BWAV")

            
        # check if there are any other offsets pointing to the replaced bwav, if there is - move offsets
        old_offset = self.asset_offsets[idx]
        same_offset_indexes = [idx_offset for idx_offset, offset in enumerate(self.asset_offsets) if old_offset == offset and idx_offset != idx]
        if same_offset_indexes:
            if not resize_if_needed:
                print(f"{name} - Replacing an asset that's referenced multiple times, that's disabled due to size differences")
                return False
            else:
                print(f"{name} - Replacing an asset that's referenced multiple times, moving offsets")

            size = bwav_old.get_size()
            idx_from = idx if same_offset_indexes[0] < idx else same_offset_indexes[0]
            for idx_resize in range(idx_from, self.asset_count):
                self.asset_offsets[idx_resize] += size # is this correct?
        
        # move offsets if there is a size difference
        size_diff = pad_till(bwav_new.get_size()) - pad_till(bwav_old.get_size())
        if size_diff != 0:
            if not resize_if_needed:
                print(f"{name} - Replacing would result in changing BARS size, which is disabled due to size differences")
                return False
            else:
                print(f"{name} - New and old BWAVs are different in size")

            for idx_resize in range(idx + 1, self.asset_count):
                self.asset_offsets[idx_resize] += size_diff

        # swap old asset with the new one
        self.assets[idx] = bwav_new

        self.size = self.get_size()
    
        return True
    
    def add_or_replace_bwav(self, bwav_path: str, resize_if_needed: bool = False) -> bool:
        import pathlib
        name = pathlib.Path(bwav_path).stem

        found = False
        for idx, meta in enumerate(self.metas):
            if meta.name == name:
                found = True
                break
        
        if found:
            self.replace_bwav(bwav_path, resize_if_needed)
            return False
        
        bwav_new = Bwav(bwav_path)
        
        # Create amta
        new_amta = Amta(None)
        
        # Create AMTA section
        new_amta.magic = b'AMTA'
        new_amta.bom = '<'
        new_amta.version_minor = 0
        new_amta.version_major = 5
        new_amta.empty_offset = 0
        new_amta.UNKNOWN_offset = 52
        new_amta.UNKNOWN2_offset = 0
        new_amta.MINF_offset = 0
        new_amta.STRINGS_offset = 0
        new_amta.empty_offset_2 = 0
        
        new_amta.name = name
        
        # Create Data Section
        new_amta.DATA_size = 40
        new_amta.name_crc = calculate_crc32_hash(name)
        new_amta.flags = 2
        new_amta.tracks_per_channel = 1
        new_amta.channel_count = 1
        new_amta.rest_of_data = b'\x00\x04'
        
        # Create Unknown section
        new_amta.UNKNOWN_section = AmtaUnknownSection(None, None)
        new_amta.UNKNOWN_section.unk_1 = 79
        new_amta.UNKNOWN_section.unk_2 = 0.190277099609375
        new_amta.UNKNOWN_section.unk_3 = 0.00570450210943818
        new_amta.UNKNOWN_section.unk_4 = -43.5840492248535
        new_amta.UNKNOWN_section.unk_5 = -43.5840492248535
        new_amta.UNKNOWN_section.unk_6 = 0.0
        
        # Convert unknown section to bytes
        new_amta.rest_of_data = new_amta.rest_of_data + new_amta.UNKNOWN_section.to_bytes(self.bom)
        
        # End of amta
        new_amta.rest_of_file = pad_to_4_byte_boundary(name.encode() + b'\x00')
        
        new_amta.size = new_amta.get_size()
        
        # Make space for header, 8 bytes per asset
        for idx, offset in enumerate(self.meta_offsets):
            self.meta_offsets[idx] = offset + 12
        
        if self.asset_count > 0:
            self.meta_offsets.append(self.meta_offsets[-1] + self.metas[-1].get_size())
        else:
            self.meta_offsets.append(self.get_size())
        
        # Add to metas before adding assets
        self.metas.append(new_amta)
        
        # Correct Asset offsets for new amta
        if self.asset_count > 0:
            offset_difference = (self.meta_offsets[-1] + self.metas[-1].get_size()) - self.asset_offsets[0]
        else:
            offset_difference = 0
        
        for idx, offset in enumerate(self.asset_offsets):
            self.asset_offsets[idx] = offset + offset_difference
        
        if self.asset_count > 0:
            self.asset_offsets.append(self.asset_offsets[-1] + self.assets[-1].get_size())
        else:
            self.asset_offsets.append(self.get_size())
        
        self.assets.append(bwav_new)
        
        self.crc_hashes.append(calculate_crc32_hash(name))

        self.asset_count += 1
        self.size = self.get_size()
    
        return True
        
# written by NanobotZ