# Written by NanobotZ
# Modified by MediaMoots and NanobotZ

import bisect
import struct
import pathlib
import binascii
from typing import List, Tuple, Union
from io import BufferedReader, BufferedWriter, IOBase

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
        return pad_till(4 + 2 + 6 + 24 + 14 + len(self.rest_of_data) + len(self.rest_of_file), 4) # one element per write
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
            self.history_sample_1, self.history_sample_2, self.padding = struct.unpack(bom + '5IH2hH', reader.read(28))
        self.is_looping = is_looping == 1

    def write(self, writer: BufferedWriter, bom: str) -> None:
        writer.write(struct.pack(bom + '2H3I', self.codec, self.channel_pan, self.sample_rate, self.num_samples_nonprefetch, self.num_samples_this)) # 16
        writer.write(self.dsp_adpcm_coefficients) # TODO write with BOM!!!
        writer.write(struct.pack(bom + '5IH2hH', self.absolute_start_samples_nonprefetch, self.absolute_start_samples_this, \
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

        for channel in self.channel_infos:
            reader.seek(pos + channel.absolute_start_samples_this)

            if channel.codec == 2:
                reader.seek(36, 1)
                samples_size = struct.unpack(self.header.bom + "I", reader.read(4))[0] + 40
                reader.seek(-40, 1)
            else:
                samples_size = size - self.channel_infos[-1].absolute_start_samples_this
            
            self.channel_samples.append(reader.read(samples_size) if samples_size > 0 else b'')

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
        return (pad_till(header_and_info_part) + samples_part) if samples_part > 0 else header_and_info_part
            
    def get_peak_volume(self) -> float:
            """returns peak volume of all channels as linear (0 to 1, both inclusive)

            used for populating AMTA"""
            decoded = self.decode()
            all_samples = [sample for channel_samples in decoded for sample in channel_samples]
            max_sample = max(abs(sample) for sample in all_samples)
            return max_sample / 32768
    
    def convert_to_prefetch(self) -> bool:
        if self.header.is_prefetch:
            return True
        
        codec_samples = [0x1000, 0x3800, 0x9000]
        codec_bytes = [0x2000, 0x2000, 0x12200]

        converted = False
        for idx, channel in enumerate(self.channel_infos):
            req_samples = codec_samples[channel.codec]
            req_bytes = codec_bytes[channel.codec]

            samples = self.channel_samples[idx]
            if channel.codec == 2:
                decoded_samples = self.decode_channel(idx, req_bytes // 2)
                format = f"{self.header.bom}{len(decoded_samples)}h"
                samples = struct.pack(format, *decoded_samples) # conversion to PCM bytes

            if channel.num_samples_this < req_samples and len(samples) < req_bytes:
                continue
            
            channel.num_samples_this = req_samples
            self.channel_samples[idx] = samples[:req_bytes]
            channel.absolute_start_samples_this = self.channel_infos[0].absolute_start_samples_this + (idx * req_bytes)
            converted = True

        if converted:
            self.header.is_prefetch = True

        return converted
    
    def decode_channel(self, channel_idx: int, sample_limit: int = 0) -> List[int]:
        """returns a list of PCM16 (short) samples"""
        assert channel_idx < self.header.num_channels

        if self.decoded_channels[channel_idx]:
            samples = self.decoded_channels[channel_idx]
            if sample_limit > 0:
                return samples[:sample_limit]
            return samples
        
        result_byte_limit = sample_limit * 2
        reduced = False

        src = self.channel_samples[channel_idx]
        dst: List[int] = []

        channel_info = self.channel_infos[channel_idx]
        if channel_info.codec == 0:
            if result_byte_limit > 0 and len(src) > result_byte_limit:
                reduced = True
                src = src[:result_byte_limit]
            format = f'{self.header.bom}{len(src) // 2}h'
            dst.extend(struct.unpack(format, src))
            # for i in range(channel_info.num_samples_this):
            #     dst.append(*struct.unpack(self.header.bom + "h", src[i*2:i*2+2]))
        elif channel_info.codec == 1: # based on https://github.com/Thealexbarney/DspTool/blob/master/dsptool/decode.c
            num_samples = channel_info.num_samples_this
            hist1 = channel_info.history_sample_1
            hist2 = channel_info.history_sample_2
            coefs: Tuple[int] = struct.unpack(self.header.bom + '16h', channel_info.dsp_adpcm_coefficients)

            SAMPLES_PER_FRAME = 14
            frame_count = (num_samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
            samples_remaining = num_samples

            dst = [0] * num_samples

            idx_src = 0
            idx_dst = 0
            for _ in range(frame_count):
                predictor = get_high_nibble(src[idx_src])
                scale = 1 << get_low_nibble(src[idx_src])
                idx_src += 1
                coef1 = coefs[predictor * 2]
                coef2 = coefs[predictor * 2 + 1]

                samples_to_read = min(SAMPLES_PER_FRAME, samples_remaining)
                even = True
                for _ in range(samples_to_read):
                    sample = 0
                    if even:
                        sample = get_high_nibble(src[idx_src])
                    else:
                        sample = get_low_nibble(src[idx_src])
                        idx_src += 1
                    even = not even
                    sample = sample - 16 if sample >= 8 else sample
                    sample = (((scale * sample) << 11) + 1024 + (coef1 * hist1 + coef2 * hist2)) >> 11

                    final_sample = sample
                    if final_sample > 32767: # short max val
                        final_sample = 32767
                    elif final_sample < -32768: # short min val
                        final_sample = -32768

                    hist2 = hist1
                    hist1 = final_sample
                    dst[idx_dst] = final_sample
                    idx_dst += 1
                
                if result_byte_limit > 0 and idx_dst > result_byte_limit:
                    reduced = True
                    dst = dst[:sample_limit]
                    break
                
                samples_remaining -= samples_to_read
        elif channel_info.codec == 2:
            import pyogg
            opus_dec = pyogg.OpusDecoder()
            opus_dec.set_channels(1)
            opus_dec.set_sampling_frequency(channel_info.sample_rate)

            data_len = len(src)
            cur_idx = 40
            total_pcm = bytes()
            while cur_idx < data_len:
                packet_size, = struct.unpack('>I', src[cur_idx:cur_idx+4])
                cur_idx += 8 # skipping 4 unknown bytes
                to_read = cur_idx + packet_size
                packet = src[cur_idx:to_read]
                decoded_pcm = opus_dec.decode(bytearray(packet))
                total_pcm += decoded_pcm
                
                cur_idx += packet_size

                if result_byte_limit > 0 and len(total_pcm) > result_byte_limit:
                    reduced = True
                    total_pcm = total_pcm[:result_byte_limit]
                    break

            format = f'{self.header.bom}{len(total_pcm) // 2}h'
            dst.extend(struct.unpack(format, total_pcm))

        if not reduced: # don't cache partially decoded samples
            self.decoded_channels[channel_idx] = dst
        
        return dst
    
    def decode(self, sample_limit_per_channel: int = 0) -> List[List[int]]:
        """returns a list of lists of PCM16 (short) samples, one list per channel"""
        result: List[List[int]] = []
        for channel_idx in range(self.header.num_channels):
            result.append(self.decode_channel(channel_idx, sample_limit_per_channel))

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

    def convert_to_opus(self) -> None:
        non_compatible = [x for x in self.channel_infos if x.codec == 2]
        if non_compatible:
            raise ValueError("Bwav is already opus")
        
        if self.header.is_prefetch:
            raise ValueError("Can't convert a prefetch")
        
        non_compatible = [x for x in self.channel_infos if x.sample_rate != 48000]
        if non_compatible:
            raise ValueError("Invalid sample rate, must be 48000Hz")
        
        import pyogg

        converted: List[bytes] = []
        for channel_idx in range(self.header.num_channels):
            samples = self.decode_channel(channel_idx)

            pack_format = f"{self.header.bom}{len(samples)}h"
            data = struct.pack(pack_format, *samples)
            data_len = len(data)

            opus_enc = pyogg.OpusEncoder()
            opus_enc.set_application("audio")
            opus_enc.set_sampling_frequency(self.channel_infos[channel_idx].sample_rate)
            opus_enc.set_channels(1)
            opus_enc.set_max_bytes_per_frame(240)

            desired_frame_duration = 20/1000 # 20 milliseconds
            desired_frame_size = int(desired_frame_duration * self.channel_infos[channel_idx].sample_rate)
            bits_per_frame = 2
            desired_data_portion_size = desired_frame_size * bits_per_frame

            cur_idx = 0
            result = bytes()
            while cur_idx < data_len:
                read_to = cur_idx + desired_data_portion_size
                pcm = data[cur_idx:read_to]

                if len(pcm) < desired_data_portion_size:
                    null_byte_count = desired_data_portion_size - len(pcm)
                    pcm += b"\x00" * null_byte_count

                encoded = opus_enc.encode(pcm)
                result += struct.pack('>I', len(encoded)) + (b'\x00' * 4) + encoded
                cur_idx += desired_data_portion_size

            skip = opus_enc.get_algorithmic_delay()

            final_opus = b'\x01\x00\x00\x80\x18\x00\x00\x00\x00\x01\x00\x00\x80\xBB\x00\x00\x20\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            final_opus += struct.pack(self.header.bom + "I", skip)
            final_opus += b'\x04\x00\x00\x80'
            final_opus += struct.pack(self.header.bom + "I", len(result))
            final_opus += result
            
            converted.append(final_opus)
        
        initial_offset = self.channel_infos[0].absolute_start_samples_this
        for channel_idx, channel_info in enumerate(self.channel_infos):
            channel_info.codec = 2
            channel_info.dsp_adpcm_coefficients = b'\x00' * 32
            channel_info.history_sample_1 = 0
            channel_info.history_sample_2 = 0
            channel_info.predictor_scale = 0

            final_opus = converted[channel_idx]
            final_size = len(final_opus)
            self.channel_samples[channel_idx] = final_opus

            channel_info.absolute_start_samples_this = initial_offset
            channel_info.absolute_start_samples_nonprefetch = initial_offset

            # not sure if these are required - all ToTK opus bwavs had these
            channel_info.is_looping = True
            channel_info.loop_start_sample = 0x00
            channel_info.loop_end_sample = 0xFFFFFFFF

            initial_offset += pad_till(final_size)
        
        self.recalculate_crc()
    
    def recalculate_crc(self) -> None:
        crc = 0
        for sample_data in self.channel_samples:
            crc = binascii.crc32(sample_data, crc)
        self.header.crc = crc
    
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
        self.meta_count: int
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

        meta_count = reader.read(4)

        self.size, self.version_minor, self.version_major, self.meta_count = struct.unpack(self.bom + 'I2BI', size + version + meta_count)

        self.crc_hashes.extend(struct.unpack(self.bom + 'I' * self.meta_count, reader.read(4 * self.meta_count)))

        for _ in range(self.meta_count):
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
        writer.write(struct.pack(self.bom + '2BI', self.version_minor, self.version_major, self.meta_count)) # 6

        writer.write(struct.pack(self.bom + 'I' * self.meta_count, *self.crc_hashes)) # 4 * self.meta_count

        for idx in range(self.meta_count):
            writer.write(struct.pack(self.bom + '2I', self.meta_offsets[idx], self.asset_offsets[idx])) # 8 * self.meta_count

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
        header_crc_metas_part = self.get_header_size(self.meta_count)

        # get only unique assets, as some metas can point to the same asset
        unique_assets: List[Tuple[int, int]] = []
        for idx in range(self.meta_count):
            if not [idx_offset_tuple for idx_offset_tuple in unique_assets if idx_offset_tuple[1] == self.asset_offsets[idx]]:
                unique_assets.append((idx, self.asset_offsets[idx]))
            
        assets_part = 0  
        if self.meta_count > 0:
            last_idx = unique_assets[-1][0]

            # condition below - the last BWAV doesn't need to be padded
            assets_part = sum([pad_till(self.assets[idx].get_size()) if idx != last_idx else self.assets[idx].get_size() for idx, _ in unique_assets])

        full_size = header_crc_metas_part + assets_part

        return full_size
    
    def get_header_size(self, custom_count) -> int:
        return pad_till(4 + 4 + 2 + 6 + (4 * custom_count) + (8 * custom_count) + len(self.unknown) + (sum([meta.get_size() for meta in self.metas])))
    
    def get_preheader_size(self, custom_count) -> int:
        return 4 + 4 + 2 + 6 + (4 * custom_count) + (8 * custom_count) + len(self.unknown)
    
    def calculate_offsets(self):
        # Calculate meta offsets
        bars_preheader_size = self.get_preheader_size(self.meta_count)
        self.meta_offsets.clear()
        for idx, meta in enumerate(self.metas):
            if idx > 0:
                self.meta_offsets.append(self.meta_offsets[-1] + meta.get_size())
            else:
                self.meta_offsets.append(bars_preheader_size)
        
        # Calculate asset offsets
        bars_header_size = self.get_header_size(self.meta_count)
        self.asset_offsets.clear()
        for idx, asset in enumerate(self.assets):
            if idx > 0:
                self.asset_offsets.append(self.asset_offsets[-1] + pad_till(self.assets[idx - 1].get_size()))
            else:
                self.asset_offsets.append(bars_header_size)
            
    
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
            
            if not bwav_new.convert_to_prefetch():
                print(f"{name} - Couldn't convert the new BWAV to prefetch...")
                return False
            else:
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
            for idx_resize in range(idx_from, self.meta_count):
                self.asset_offsets[idx_resize] += size # is this correct?
        
        # move offsets if there is a size difference
        size_diff = pad_till(bwav_new.get_size()) - pad_till(bwav_old.get_size())
        if size_diff != 0:
            if not resize_if_needed:
                print(f"{name} - Replacing would result in changing BARS size, which is disabled due to size differences")
                return False
            else:
                print(f"{name} - New and old BWAVs are different in size")

            for idx_resize in range(idx + 1, self.meta_count):
                self.asset_offsets[idx_resize] += size_diff

        # swap old asset with the new one
        self.assets[idx] = bwav_new

        self.size = self.get_size()
    
        return True
    
    def create_new_amta(self, name, bwav):
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
        new_amta.UNKNOWN_section.unk_2 = bwav.get_peak_volume()
        new_amta.UNKNOWN_section.unk_3 = 0.005
        new_amta.UNKNOWN_section.unk_4 = -43.6
        new_amta.UNKNOWN_section.unk_5 = -43.6
        new_amta.UNKNOWN_section.unk_6 = 0.0
        
        # Convert unknown section to bytes
        new_amta.rest_of_data = new_amta.rest_of_data + new_amta.UNKNOWN_section.to_bytes(self.bom)
        
        # End of amta
        new_amta.rest_of_file = name.encode() + b'\x00'
        
        new_amta.size = new_amta.get_size()
        return new_amta
    
    def add_or_replace_bwav(self, bwav_path: str, resize_if_needed: bool = False) -> bool:
        
        # Get and calculate name
        name = pathlib.Path(bwav_path).stem
        name_hash = calculate_crc32_hash(name)
        
        # Check if exists        
        if name_hash in self.crc_hashes:
            self.replace_bwav(bwav_path, resize_if_needed)
            return False
        
        # Create amta and bwav 
        new_bwav = Bwav(bwav_path)
        new_amta = self.create_new_amta(name, new_bwav)
        
        # Calculate Insertion point
        insertion_index = bisect.bisect_left(self.crc_hashes, name_hash)
        
        # Add to lists
        self.metas.insert(insertion_index, new_amta)
        self.assets.insert(insertion_index, new_bwav)
        self.crc_hashes.insert(insertion_index, name_hash)

        # Set bars vars
        self.meta_count += 1
        self.calculate_offsets()
        self.size = self.get_size()
    
        return True
        
# Written by NanobotZ
# Modified by MediaMoots and NanobotZ