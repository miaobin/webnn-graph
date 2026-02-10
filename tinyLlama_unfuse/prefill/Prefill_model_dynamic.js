/**
 * Helper class for loading and managing WebNN graph weights
 */
export class WeightsFile {
  constructor(buffer, manifest) {
    this.buffer = buffer;
    this.manifest = manifest;
  }

  /**
   * Load weights from URL paths
   * @param {string} weightsPath - Path to .weights binary file
   * @param {string} manifestPath - Path to .manifest.json file
   * @returns {Promise<WeightsFile>}
   */
  static async load(weightsPath, manifestPath) {
    const [weightsResponse, manifestResponse] = await Promise.all([
      fetch(weightsPath),
      fetch(manifestPath)
    ]);

    if (!weightsResponse.ok) {
      throw new Error(`Failed to load weights: ${weightsResponse.statusText}`);
    }
    if (!manifestResponse.ok) {
      throw new Error(`Failed to load manifest: ${manifestResponse.statusText}`);
    }

    const buffer = await weightsResponse.arrayBuffer();
    const manifest = await manifestResponse.json();

    // Validate manifest format
    if (manifest.format !== 'wg-weights-manifest') {
      throw new Error(`Invalid manifest format: ${manifest.format}`);
    }
    if (manifest.version !== 1) {
      throw new Error(`Unsupported manifest version: ${manifest.version}`);
    }

    // Validate weights file header
    const view = new DataView(buffer);
    const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
    if (magic !== 'WGWT') {
      throw new Error(`Invalid weights file magic: ${magic}`);
    }
    const version = view.getUint32(4, true); // little-endian
    if (version !== 1) {
      throw new Error(`Unsupported weights file version: ${version}`);
    }

    return new WeightsFile(buffer, manifest);
  }

  /**
   * Get a slice descriptor for a named tensor
   * @param {string} name - Tensor name
   * @returns {Object} Tensor metadata with byteOffset and byteLength
   */
  getSlice(name) {
    const tensor = this.manifest.tensors[name];
    if (!tensor) {
      throw new Error(`Tensor not found in manifest: ${name}`);
    }
    return tensor;
  }

  /**
   * Get the raw data for a named tensor
   * @param {string} name - Tensor name
   * @returns {ArrayBuffer} Tensor data
   */
  getData(name) {
    const tensor = this.getSlice(name);
    return this.buffer.slice(tensor.byteOffset, tensor.byteOffset + tensor.byteLength);
  }

  /**
   * List all available tensor names
   * @returns {string[]}
   */
  getTensorNames() {
    return Object.keys(this.manifest.tensors);
  }
}

/**
 * Build a WebNN MLGraph from the graph definition
 * @param {MLContext} context - WebNN context
 * @param {WeightsFile} weights - Loaded weights file
 * @returns {Promise<MLGraph>}
 */
export async function buildGraph(context, weights, max_sequence_length) {
  const builder = new MLGraphBuilder(context);
  const env = new Map();

  const sequence_length = {name: 'sequence_length', maxSize: 2048};
  const total_sequence_length = {name: 'total_sequence_length', maxSize: 2048};
  const past_sequence_length = max_sequence_length;

  env.set("attention_mask_593", builder.input("attention_mask_593", { dataType: "int64", shape: [1, total_sequence_length] }));
  env.set("input_ids_580", builder.input("input_ids_580", { dataType: "int64", shape: [1, sequence_length] }));
  env.set("past_key_values_0_key_649", builder.input("past_key_values_0_key_649", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_0_value_606", builder.input("past_key_values_0_value_606", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_10_key_1740", builder.input("past_key_values_10_key_1740", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_10_value_1704", builder.input("past_key_values_10_value_1704", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_11_key_1849", builder.input("past_key_values_11_key_1849", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_11_value_1813", builder.input("past_key_values_11_value_1813", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_12_key_1958", builder.input("past_key_values_12_key_1958", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_12_value_1922", builder.input("past_key_values_12_value_1922", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_13_key_2067", builder.input("past_key_values_13_key_2067", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_13_value_2031", builder.input("past_key_values_13_value_2031", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_14_key_2176", builder.input("past_key_values_14_key_2176", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_14_value_2140", builder.input("past_key_values_14_value_2140", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_15_key_2285", builder.input("past_key_values_15_key_2285", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_15_value_2249", builder.input("past_key_values_15_value_2249", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_16_key_2394", builder.input("past_key_values_16_key_2394", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_16_value_2358", builder.input("past_key_values_16_value_2358", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_17_key_2503", builder.input("past_key_values_17_key_2503", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_17_value_2467", builder.input("past_key_values_17_value_2467", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_18_key_2612", builder.input("past_key_values_18_key_2612", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_18_value_2576", builder.input("past_key_values_18_value_2576", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_19_key_2721", builder.input("past_key_values_19_key_2721", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_19_value_2685", builder.input("past_key_values_19_value_2685", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_1_key_759", builder.input("past_key_values_1_key_759", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_1_value_723", builder.input("past_key_values_1_value_723", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_20_key_2830", builder.input("past_key_values_20_key_2830", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_20_value_2794", builder.input("past_key_values_20_value_2794", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_21_key_2939", builder.input("past_key_values_21_key_2939", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_21_value_2903", builder.input("past_key_values_21_value_2903", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_2_key_868", builder.input("past_key_values_2_key_868", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_2_value_832", builder.input("past_key_values_2_value_832", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_3_key_977", builder.input("past_key_values_3_key_977", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_3_value_941", builder.input("past_key_values_3_value_941", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_4_key_1086", builder.input("past_key_values_4_key_1086", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_4_value_1050", builder.input("past_key_values_4_value_1050", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_5_key_1195", builder.input("past_key_values_5_key_1195", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_5_value_1159", builder.input("past_key_values_5_value_1159", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_6_key_1304", builder.input("past_key_values_6_key_1304", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_6_value_1268", builder.input("past_key_values_6_value_1268", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_7_key_1413", builder.input("past_key_values_7_key_1413", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_7_value_1377", builder.input("past_key_values_7_value_1377", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_8_key_1522", builder.input("past_key_values_8_key_1522", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_8_value_1486", builder.input("past_key_values_8_value_1486", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_9_key_1631", builder.input("past_key_values_9_key_1631", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("past_key_values_9_value_1595", builder.input("past_key_values_9_value_1595", { dataType: "float16", shape: [1, 4, past_sequence_length, 64] }));
  env.set("position_ids_622", builder.input("position_ids_622", { dataType: "int64", shape: [1, sequence_length] }));

  {
    const sl = weights.getSlice("Inserted_1");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_10");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_10", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1001");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1001", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1003");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1003", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1004");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1004", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1005");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1005", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1006");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1006", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1007");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1007", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1008");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1008", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_101");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_101", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1012");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1012", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1013");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1013", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1014");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1014", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1015");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1015", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1016");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1016", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1017");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1017", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1018");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1018", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1019");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1019", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1021");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1021", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1022");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1022", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1023");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1023", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1024");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1024", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1025");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1025", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1026");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1026", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1028");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1028", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_103");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_103", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1030");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1030", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1032");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1032", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1033");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1033", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1034");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1034", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1036");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1036", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1037");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1037", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1038");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1038", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1039");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1039", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_104");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_104", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1040");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1040", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1041");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1041", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1042");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1042", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1043");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1043", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1045");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1045", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1046");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1046", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1047");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1047", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1048");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1048", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1049");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1049", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_105");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_105", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1050");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1050", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1051");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1051", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1052");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1052", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1053");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1053", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1054");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1054", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1055");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1055", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1057");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1057", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1059");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1059", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_106");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_106", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1061");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1061", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1063");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1063", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1064");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1064", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1065");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1065", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1066");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1066", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1067");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1067", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1068");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1068", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_107");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_107", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1072");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1072", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1073");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1073", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1074");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1074", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1075");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1075", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1076");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1076", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1077");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1077", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1078");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1078", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1079");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1079", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_108");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_108", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1081");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1081", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1082");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1082", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1083");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1083", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1084");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1084", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1085");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1085", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1086");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1086", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1088");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1088", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1090");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1090", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1092");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1092", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1093");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1093", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1094");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1094", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1096");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1096", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1097");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1097", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1098");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1098", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1099");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1099", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_11");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_11", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1100");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1100", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1101");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1101", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1102");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1102", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1103");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1103", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1105");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1105", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1106");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1106", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1107");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1107", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1108");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1108", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1109");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1109", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1110");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1110", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1111");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1111", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1112");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1112", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1113");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1113", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1114");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1114", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1115");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1115", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1117");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1117", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1119");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1119", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_112");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_112", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1121");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1121", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1123");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1123", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1124");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1124", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1125");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1125", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1126");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1126", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1127");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1127", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1128");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1128", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_113");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_113", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1132");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1132", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1133");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1133", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1134");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1134", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1135");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1135", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1136");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1136", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1137");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1137", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1138");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1138", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1139");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1139", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_114");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_114", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1141");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1141", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1142");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1142", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1143");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1143", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1144");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1144", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1145");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1145", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1146");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1146", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1148");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1148", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_115");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_115", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1150");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1150", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1152");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1152", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1153");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1153", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1154");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1154", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1156");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1156", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1157");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1157", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1158");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1158", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1159");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1159", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_116");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_116", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1160");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1160", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1161");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1161", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1162");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1162", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1163");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1163", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1165");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1165", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1166");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1166", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1167");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1167", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1168");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1168", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1169");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1169", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_117");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_117", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1170");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1170", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1171");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1171", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1172");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1172", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1173");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1173", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1174");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1174", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1175");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1175", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1177");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1177", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1179");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1179", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_118");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_118", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1181");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1181", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1183");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1183", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1184");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1184", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1185");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1185", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1186");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1186", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1187");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1187", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1188");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1188", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_119");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_119", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1192");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1192", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1193");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1193", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1194");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1194", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1195");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1195", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1196");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1196", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1197");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1197", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1198");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1198", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1199");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1199", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_12");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_12", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1201");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1201", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1202");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1202", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1203");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1203", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1204");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1204", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1205");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1205", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1206");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1206", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1208");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1208", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_121");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_121", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1210");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1210", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1212");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1212", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1213");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1213", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1214");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1214", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1216");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1216", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1217");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1217", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1218");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1218", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1219");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1219", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_122");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_122", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1220");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1220", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1221");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1221", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1222");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1222", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1223");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1223", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1225");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1225", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1226");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1226", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1227");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1227", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1228");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1228", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1229");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1229", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_123");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_123", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1230");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1230", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1231");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1231", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1232");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1232", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1233");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1233", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1234");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1234", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1235");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1235", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1237");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1237", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1239");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1239", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_124");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_124", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1241");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1241", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1243");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1243", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1244");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1244", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1245");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1245", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1246");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1246", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1247");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1247", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1248");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1248", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_125");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_125", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1252");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1252", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1253");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1253", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1254");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1254", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1255");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1255", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1256");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1256", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1257");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1257", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1258");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1258", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1259");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1259", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_126");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_126", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1261");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1261", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1262");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1262", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1263");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1263", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1264");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1264", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1265");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1265", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1266");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1266", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1268");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1268", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1270");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1270", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1272");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1272", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1273");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1273", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1274");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1274", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1276");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1276", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1277");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1277", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1278");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1278", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1279");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1279", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_128");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_128", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1280");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1280", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1281");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1281", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1282");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1282", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1283");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1283", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1285");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1285", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1286");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1286", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1287");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1287", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1288");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1288", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1289");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1289", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1290");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1290", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1291");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1291", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1292");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1292", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1293");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1293", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1294");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1294", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1295");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1295", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1297");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1297", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1299");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1299", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_13");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_13", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_130");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_130", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1301");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1301", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1303");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1303", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1304");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1304", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1305");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1305", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1306");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1306", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1307");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1307", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1308");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1308", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1312");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1312", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1313");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1313", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1314");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1314", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1315");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1315", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1316");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1316", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1317");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1317", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1318");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1318", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1319");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1319", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_132");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_132", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1321");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1321", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1322");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1322", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1323");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1323", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1324");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1324", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1325");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1325", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1326");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1326", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1328");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1328", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_133");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_133", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1330");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1330", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1332");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1332", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1333");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1333", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1334");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1334", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1336");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1336", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1337");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1337", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1338");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1338", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1339");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1339", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_134");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_134", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1340");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1340", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1341");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1341", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1342");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1342", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1343");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1343", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1345");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1345", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1346");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1346", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1347");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1347", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1348");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1348", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1349");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1349", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1350");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1350", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1351");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1351", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1352");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1352", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1353");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1353", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1354");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1354", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1355");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1355", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1357");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1357", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1359");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1359", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_136");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_136", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1361");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1361", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1363");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1363", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1364");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1364", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1365");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1365", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1366");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1366", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1367");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1367", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1368");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1368", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_137");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_137", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1372");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1372", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1373");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1373", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1374");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1374", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1375");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1375", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1376");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1376", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1377");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1377", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1378");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1378", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1379");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1379", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_138");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_138", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1381");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1381", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1382");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1382", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1383");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1383", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1384");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1384", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1385");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1385", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1386");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1386", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1388");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1388", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_139");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_139", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1390");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1390", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1392");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1392", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1393");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1393", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1394");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1394", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1396");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1396", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1397");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1397", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1398");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1398", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1399");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1399", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_14");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_14", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_140");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_140", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1400");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1400", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1401");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1401", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1402");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1402", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1403");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1403", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1405");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1405", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1406");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1406", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1407");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1407", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1408");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1408", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1409");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1409", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_141");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_141", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1410");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1410", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1411");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1411", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1412");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1412", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1413");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1413", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1414");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1414", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_142");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_142", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_143");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_143", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_145");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_145", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_146");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_146", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_147");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_147", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_148");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_148", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_149");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_149", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_15");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_15", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_150");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_150", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_151");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_151", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_152");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_152", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_153");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_153", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_154");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_154", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_155");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_155", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_157");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_157", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_159");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_159", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_16");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_16", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_161");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_161", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_163");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_163", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_164");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_164", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_165");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_165", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_166");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_166", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_167");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_167", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_168");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_168", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_17");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_17", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_172");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_172", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_173");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_173", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_174");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_174", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_175");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_175", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_176");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_176", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_177");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_177", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_178");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_178", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_179");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_179", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_18");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_18", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_181");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_181", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_182");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_182", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_183");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_183", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_184");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_184", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_185");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_185", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_186");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_186", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_188");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_188", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_19");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_19", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_190");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_190", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_192");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_192", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_193");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_193", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_194");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_194", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_196");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_196", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_197");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_197", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_198");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_198", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_199");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_199", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_2");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_2", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_20");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_20", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_200");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_200", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_201");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_201", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_202");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_202", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_203");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_203", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_205");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_205", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_206");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_206", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_207");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_207", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_208");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_208", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_209");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_209", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_21");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_21", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_210");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_210", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_211");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_211", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_212");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_212", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_213");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_213", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_214");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_214", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_215");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_215", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_217");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_217", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_219");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_219", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_22");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_22", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_221");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_221", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_223");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_223", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_224");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_224", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_225");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_225", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_226");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_226", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_227");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_227", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_228");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_228", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_23");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_23", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_232");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_232", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_233");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_233", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_234");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_234", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_235");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_235", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_236");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_236", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_237");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_237", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_238");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_238", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_239");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_239", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_24");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_24", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_241");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_241", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_242");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_242", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_243");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_243", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_244");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_244", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_245");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_245", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_246");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_246", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_248");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_248", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_25");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_25", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_250");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_250", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_252");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_252", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_253");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_253", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_254");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_254", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_256");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_256", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_257");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_257", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_258");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_258", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_259");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_259", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_26");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_26", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_260");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_260", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_261");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_261", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_262");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_262", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_263");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_263", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_265");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_265", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_266");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_266", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_267");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_267", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_268");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_268", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_269");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_269", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_27");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_27", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_270");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_270", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_271");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_271", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_272");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_272", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_273");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_273", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_274");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_274", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_275");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_275", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_277");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_277", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_279");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_279", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_28");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_28", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_281");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_281", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_283");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_283", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_284");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_284", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_285");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_285", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_286");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_286", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_287");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_287", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_288");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_288", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_29");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_29", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_292");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_292", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_293");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_293", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_294");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_294", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_295");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_295", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_296");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_296", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_297");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_297", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_298");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_298", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_299");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_299", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_3");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_3", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_30");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_30", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_301");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_301", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_302");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_302", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_303");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_303", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_304");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_304", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_305");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_305", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_306");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_306", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_308");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_308", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_31");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_31", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_310");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_310", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_312");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_312", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_313");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_313", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_314");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_314", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_316");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_316", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_317");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_317", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_318");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_318", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_319");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_319", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_32");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_32", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_320");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_320", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_321");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_321", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_322");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_322", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_323");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_323", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_325");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_325", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_326");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_326", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_327");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_327", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_328");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_328", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_329");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_329", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_33");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_33", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_330");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_330", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_331");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_331", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_332");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_332", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_333");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_333", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_334");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_334", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_335");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_335", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_337");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_337", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_339");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_339", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_34");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_34", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_341");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_341", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_343");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_343", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_344");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_344", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_345");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_345", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_346");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_346", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_347");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_347", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_348");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_348", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_35");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_35", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_352");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_352", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_353");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_353", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_354");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_354", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_355");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_355", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_356");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_356", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_357");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_357", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_358");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_358", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_359");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_359", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_36");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_36", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_361");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_361", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_362");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_362", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_363");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_363", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_364");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_364", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_365");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_365", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_366");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_366", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_368");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_368", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_37");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_37", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_370");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_370", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_372");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_372", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_373");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_373", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_374");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_374", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_376");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_376", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_377");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_377", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_378");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_378", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_379");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_379", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_38");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_38", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_380");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_380", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_381");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_381", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_382");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_382", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_383");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_383", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_385");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_385", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_386");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_386", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_387");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_387", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_388");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_388", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_389");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_389", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_39");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_39", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_390");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_390", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_391");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_391", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_392");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_392", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_393");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_393", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_394");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_394", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_395");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_395", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_397");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_397", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_399");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_399", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_4");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_4", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_40");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_40", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_401");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_401", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_403");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_403", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_404");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_404", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_405");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_405", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_406");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_406", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_407");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_407", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_408");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_408", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_41");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_41", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_412");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_412", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_413");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_413", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_414");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_414", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_415");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_415", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_416");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_416", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_417");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_417", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_418");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_418", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_419");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_419", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_42");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_42", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_421");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_421", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_422");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_422", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_423");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_423", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_424");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_424", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_425");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_425", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_426");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_426", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_428");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_428", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_43");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_43", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_430");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_430", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_432");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_432", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_433");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_433", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_434");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_434", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_436");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_436", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_437");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_437", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_438");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_438", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_439");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_439", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_44");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_44", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_440");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_440", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_441");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_441", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_442");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_442", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_443");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_443", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_445");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_445", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_446");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_446", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_447");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_447", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_448");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_448", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_449");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_449", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_45");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_45", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_450");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_450", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_451");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_451", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_452");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_452", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_453");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_453", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_454");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_454", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_455");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_455", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_457");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_457", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_459");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_459", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_46");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_46", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_461");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_461", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_463");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_463", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_464");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_464", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_465");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_465", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_466");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_466", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_467");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_467", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_468");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_468", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_47");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_47", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_472");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_472", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_473");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_473", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_474");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_474", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_475");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_475", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_476");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_476", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_477");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_477", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_478");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_478", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_479");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_479", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_48");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_48", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_481");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_481", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_482");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_482", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_483");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_483", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_484");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_484", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_485");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_485", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_486");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_486", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_488");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_488", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_49");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_49", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_490");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_490", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_492");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_492", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_493");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_493", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_494");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_494", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_496");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_496", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_497");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_497", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_498");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_498", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_499");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_499", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_5");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_5", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_50");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_50", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_500");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_500", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_501");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_501", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_502");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_502", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_503");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_503", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_505");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_505", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_506");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_506", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_507");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_507", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_508");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_508", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_509");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_509", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_51");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_51", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_510");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_510", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_511");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_511", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_512");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_512", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_513");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_513", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_514");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_514", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_515");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_515", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_517");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_517", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_519");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_519", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_52");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_52", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_521");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_521", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_523");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_523", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_524");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_524", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_525");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_525", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_526");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_526", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_527");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_527", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_528");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_528", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_53");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_53", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_532");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_532", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_533");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_533", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_534");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_534", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_535");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_535", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_536");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_536", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_537");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_537", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_538");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_538", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_539");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_539", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_54");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_54", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_541");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_541", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_542");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_542", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_543");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_543", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_544");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_544", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_545");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_545", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_546");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_546", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_548");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_548", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_55");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_55", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_550");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_550", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_552");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_552", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_553");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_553", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_554");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_554", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_556");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_556", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_557");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_557", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_558");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_558", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_559");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_559", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_56");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_56", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_560");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_560", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_561");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_561", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_562");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_562", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_563");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_563", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_565");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_565", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_566");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_566", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_567");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_567", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_568");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_568", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_569");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_569", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_57");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_57", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_570");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_570", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_571");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_571", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_572");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_572", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_573");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_573", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_574");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_574", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_575");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_575", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_577");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_577", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_579");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_579", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_58");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_58", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_581");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_581", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_583");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_583", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_584");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_584", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_585");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_585", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_586");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_586", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_587");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_587", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_588");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_588", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_59");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_59", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_592");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_592", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_593");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_593", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_594");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_594", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_595");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_595", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_596");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_596", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_597");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_597", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_598");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_598", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_599");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_599", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_6");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_6", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_60");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_60", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_601");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_601", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_602");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_602", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_603");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_603", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_604");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_604", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_605");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_605", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_606");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_606", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_608");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_608", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_61");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_61", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_610");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_610", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_612");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_612", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_613");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_613", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_614");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_614", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_616");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_616", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_617");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_617", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_618");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_618", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_619");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_619", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_62");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_62", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_620");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_620", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_621");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_621", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_622");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_622", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_623");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_623", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_625");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_625", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_626");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_626", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_627");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_627", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_628");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_628", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_629");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_629", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_63");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_63", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_630");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_630", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_631");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_631", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_632");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_632", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_633");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_633", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_634");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_634", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_635");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_635", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_637");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_637", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_639");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_639", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_64");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_64", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_641");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_641", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_643");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_643", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_644");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_644", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_645");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_645", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_646");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_646", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_647");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_647", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_648");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_648", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_65");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_65", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_652");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_652", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_653");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_653", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_654");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_654", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_655");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_655", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_656");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_656", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_657");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_657", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_658");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_658", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_659");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_659", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_66");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_66", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_661");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_661", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_662");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_662", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_663");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_663", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_664");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_664", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_665");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_665", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_666");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_666", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_668");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_668", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_67");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_67", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_670");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_670", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_672");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_672", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_673");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_673", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_674");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_674", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_676");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_676", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_677");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_677", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_678");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_678", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_679");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_679", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_68");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_68", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_680");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_680", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_681");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_681", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_682");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_682", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_683");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_683", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_685");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_685", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_686");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_686", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_687");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_687", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_688");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_688", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_689");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_689", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_69");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_69", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_690");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_690", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_691");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_691", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_692");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_692", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_693");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_693", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_694");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_694", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_695");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_695", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_697");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_697", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_699");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_699", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_7");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_7", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_70");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_70", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_701");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_701", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_703");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_703", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_704");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_704", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_705");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_705", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_706");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_706", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_707");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_707", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_708");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_708", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_71");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_71", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_712");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_712", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_713");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_713", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_714");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_714", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_715");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_715", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_716");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_716", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_717");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_717", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_718");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_718", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_719");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_719", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_72");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_72", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_721");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_721", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_722");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_722", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_723");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_723", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_724");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_724", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_725");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_725", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_726");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_726", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_728");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_728", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_73");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_73", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_730");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_730", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_732");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_732", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_733");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_733", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_734");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_734", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_736");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_736", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_737");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_737", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_738");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_738", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_739");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_739", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_74");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_74", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_740");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_740", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_741");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_741", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_742");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_742", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_743");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_743", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_745");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_745", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_746");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_746", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_747");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_747", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_748");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_748", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_749");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_749", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_75");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_75", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_750");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_750", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_751");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_751", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_752");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_752", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_753");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_753", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_754");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_754", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_755");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_755", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_757");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_757", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_759");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_759", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_76");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_76", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_761");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_761", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_763");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_763", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_764");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_764", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_765");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_765", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_766");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_766", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_767");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_767", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_768");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_768", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_77");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_77", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_772");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_772", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_773");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_773", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_774");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_774", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_775");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_775", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_776");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_776", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_777");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_777", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_778");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_778", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_779");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_779", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_78");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_78", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_781");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_781", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_782");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_782", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_783");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_783", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_784");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_784", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_785");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_785", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_786");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_786", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_788");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_788", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_79");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_79", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_790");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_790", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_792");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_792", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_793");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_793", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_794");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_794", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_796");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_796", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_797");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_797", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_798");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_798", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_799");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_799", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_8");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_8", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_80");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_80", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_800");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_800", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_801");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_801", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_802");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_802", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_803");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_803", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_805");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_805", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_806");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_806", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_807");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_807", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_808");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_808", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_809");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_809", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_81");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_81", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_810");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_810", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_811");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_811", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_812");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_812", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_813");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_813", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_814");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_814", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_815");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_815", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_817");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_817", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_819");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_819", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_82");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_82", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_821");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_821", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_823");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_823", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_824");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_824", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_825");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_825", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_826");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_826", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_827");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_827", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_828");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_828", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_83");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_83", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_832");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_832", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_833");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_833", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_834");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_834", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_835");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_835", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_836");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_836", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_837");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_837", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_838");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_838", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_839");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_839", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_84");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_84", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_841");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_841", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_842");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_842", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_843");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_843", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_844");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_844", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_845");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_845", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_846");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_846", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_848");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_848", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_85");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_85", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_850");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_850", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_852");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_852", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_853");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_853", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_854");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_854", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_856");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_856", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_857");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_857", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_858");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_858", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_859");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_859", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_86");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_86", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_860");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_860", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_861");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_861", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_862");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_862", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_863");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_863", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_865");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_865", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_866");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_866", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_867");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_867", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_868");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_868", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_869");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_869", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_87");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_87", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_870");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_870", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_871");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_871", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_872");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_872", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_873");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_873", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_874");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_874", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_875");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_875", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_877");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_877", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_879");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_879", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_88");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_88", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_881");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_881", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_883");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_883", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_884");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_884", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_885");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_885", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_886");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_886", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_887");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_887", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_888");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_888", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_89");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_89", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_892");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_892", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_893");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_893", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_894");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_894", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_895");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_895", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_896");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_896", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_897");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_897", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_898");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_898", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_899");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_899", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_9");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_9", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_901");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_901", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_902");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_902", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_903");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_903", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_904");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_904", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_905");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_905", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_906");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_906", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_908");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_908", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_91");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_91", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_910");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_910", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_912");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_912", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_913");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_913", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_914");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_914", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_916");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_916", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_917");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_917", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_918");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_918", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_919");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_919", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_92");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_92", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_920");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_920", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_921");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_921", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_922");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_922", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_923");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_923", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_925");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_925", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_926");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_926", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_927");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_927", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_928");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_928", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_929");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_929", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_93");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_93", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_930");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_930", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_931");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_931", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_932");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_932", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_933");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_933", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_934");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_934", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_935");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_935", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_937");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_937", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_939");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_939", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_94");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_94", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_941");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_941", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_943");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_943", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_944");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_944", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_945");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_945", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_946");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_946", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_947");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_947", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_948");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_948", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_95");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_95", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_952");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_952", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_953");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_953", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_954");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_954", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_955");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_955", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_956");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_956", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_957");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_957", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_958");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_958", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_959");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_959", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_961");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_961", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_962");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_962", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_963");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_963", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_964");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_964", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_965");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_965", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_966");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_966", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_968");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_968", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_97");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_97", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_970");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_970", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_972");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_972", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_973");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_973", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_974");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_974", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_976");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_976", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_977");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_977", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_978");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_978", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_979");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_979", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_980");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_980", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_981");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_981", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_982");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_982", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_983");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_983", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_985");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_985", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_986");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_986", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_987");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_987", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_988");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_988", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_989");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_989", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_99");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_99", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_990");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_990", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_991");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_991", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_992");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_992", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_993");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_993", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_994");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_994", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_995");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_995", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_997");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_997", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_999");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_999", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("_100");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_100", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_101");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_101", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1022");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1022", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1025");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1025", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1026");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1026", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1027");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1027", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1042");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1042", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_105");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_105", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_106");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_106", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1063");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1063", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1064");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1064", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1065");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1065", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_107");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_107", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1076");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1076", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1093");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1093", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1094");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1094", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1095");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1095", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1106");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1106", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_111");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_111", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_112");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_112", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_113");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_113", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1131");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1131", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1134");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1134", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1135");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1135", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1136");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1136", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1151");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1151", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_117");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_117", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1172");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1172", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1173");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1173", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1174");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1174", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_118");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_118", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1185");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1185", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_119");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_119", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1202");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1202", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1203");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1203", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1204");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1204", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1215");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1215", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_123");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_123", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_124");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_124", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1240");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1240", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1243");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1243", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1244");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1244", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1245");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1245", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_125");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_125", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1260");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1260", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1281");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1281", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1282");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1282", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1283");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1283", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_129");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_129", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1294");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1294", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_130");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_130", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_131");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_131", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1311");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1311", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1312");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1312", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1313");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1313", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1324");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1324", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1349");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1349", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_135");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_135", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1352");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1352", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1353");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1353", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1354");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1354", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_136");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_136", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1369");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1369", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_137");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_137", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1390");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1390", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1391");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1391", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1392");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1392", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1403");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1403", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_141");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_141", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_142");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_142", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1420");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1420", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1421");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1421", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1422");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1422", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_143");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_143", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1433");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1433", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1458");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1458", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1461");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1461", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1462");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1462", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1463");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1463", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_147");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_147", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1478");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1478", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_148");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_148", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_149");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_149", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1499");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1499", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1500");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1500", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1501");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1501", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1512");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1512", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1529");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1529", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_153");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_153", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1530");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1530", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1531");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1531", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_154");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_154", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1542");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1542", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_155");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_155", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1567");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1567", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1570");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1570", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1571");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1571", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1572");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1572", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1587");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1587", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_159");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_159", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_160");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_160", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1608");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1608", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1609");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1609", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_161");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_161", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1610");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1610", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1621");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1621", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1638");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1638", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1639");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1639", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1640");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1640", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_165");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_165", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1651");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1651", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_166");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_166", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_167");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_167", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1676");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1676", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1679");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1679", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1680");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1680", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1681");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1681", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1696");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1696", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_171");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_171", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1717");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1717", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1718");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1718", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1719");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1719", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_172");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_172", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_173");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_173", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1730");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1730", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1747");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1747", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1748");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1748", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1749");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1749", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1760");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1760", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_177");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_177", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_178");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_178", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1785");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1785", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1788");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1788", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1789");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1789", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_179");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_179", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1790");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1790", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1805");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1805", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1826");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1826", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1827");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1827", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1828");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1828", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_183");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_183", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1839");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1839", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_184");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_184", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_185");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_185", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1856");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1856", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1857");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1857", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1858");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1858", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1869");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1869", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_189");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_189", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1894");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1894", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1897");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1897", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1898");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1898", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1899");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1899", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_190");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_190", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_191");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_191", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1914");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1914", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1935");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1935", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1936");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1936", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1937");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1937", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1948");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1948", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_195");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_195", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_196");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_196", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1965");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1965", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1966");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1966", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1967");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1967", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_197");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_197", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1978");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1978", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2003");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2003", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2006");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2006", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2007");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2007", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2008");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2008", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_201");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_201", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_202");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_202", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2023");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2023", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_203");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_203", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2044");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2044", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2045");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2045", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2046");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2046", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2057");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2057", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_207");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_207", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2074");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2074", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2075");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2075", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2076");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2076", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_208");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_208", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2087");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2087", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_209");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_209", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2112");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2112", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2115");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2115", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2116");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2116", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2117");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2117", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_213");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_213", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2132");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2132", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_214");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_214", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_215");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_215", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2153");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2153", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2154");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2154", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2155");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2155", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2166");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2166", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2183");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2183", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2184");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2184", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2185");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2185", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_219");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_219", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2196");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2196", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_220");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_220", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_221");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_221", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2221");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2221", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2224");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2224", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2225");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2225", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2226");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2226", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2241");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2241", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_225");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_225", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_226");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_226", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2262");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2262", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2263");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2263", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2264");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2264", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_227");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_227", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2275");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2275", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2292");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2292", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2293");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2293", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2294");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2294", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2305");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2305", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_231");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_231", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_232");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_232", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_233");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_233", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2330");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2330", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2333");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2333", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2334");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2334", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2335");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2335", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2350");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2350", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_237");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_237", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2371");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2371", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2372");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2372", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2373");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2373", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_238");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_238", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2384");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2384", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_239");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_239", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2401");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2401", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2402");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2402", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2403");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2403", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2414");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2414", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_243");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_243", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2439");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2439", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_244");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_244", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2442");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2442", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2443");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2443", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2444");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2444", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_245");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_245", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2459");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2459", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2480");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2480", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2481");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2481", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2482");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2482", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_249");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_249", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2493");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2493", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_250");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_250", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_251");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_251", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2510");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2510", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2511");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2511", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2512");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2512", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2523");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2523", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2548");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2548", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_255");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_255", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2551");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2551", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2552");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2552", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2553");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2553", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_256");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_256", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2568");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2568", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_257");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_257", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2589");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2589", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2590");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2590", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2591");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2591", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2602");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2602", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_261");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_261", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2619");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2619", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_262");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_262", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2620");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2620", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2621");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2621", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_263");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_263", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2632");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2632", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2657");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2657", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2660");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2660", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2661");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2661", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2662");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2662", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_267");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_267", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2677");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2677", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_268");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_268", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_269");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_269", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2698");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2698", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2699");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2699", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2700");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2700", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2711");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2711", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2728");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2728", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2729");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2729", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_273");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_273", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2730");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2730", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_274");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_274", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2741");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2741", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_275");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_275", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2766");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2766", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2769");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2769", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2770");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2770", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2771");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2771", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2786");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2786", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_279");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_279", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_280");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_280", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2807");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2807", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2808");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2808", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2809");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2809", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_281");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_281", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2820");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2820", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2837");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2837", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2838");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2838", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2839");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2839", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_285");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_285", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2850");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2850", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_286");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_286", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_287");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_287", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2875");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2875", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2878");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2878", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2879");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2879", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2880");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2880", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2895");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2895", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_291");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_291", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2916");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2916", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2917");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2917", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2918");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2918", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_292");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_292", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2929");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2929", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_293");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_293", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2946");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2946", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2947");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2947", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2948");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2948", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2959");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2959", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_297");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_297", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_298");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_298", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2984");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2984", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2987");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2987", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2988");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2988", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2989");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2989", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_299");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_299", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3004");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3004", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_303");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_303", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_304");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_304", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_305");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_305", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_309");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_309", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_310");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_310", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_311");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_311", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_315");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_315", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_316");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_316", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_317");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_317", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_321");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_321", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_322");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_322", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_323");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_323", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_327");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_327", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_328");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_328", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_329");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_329", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_333");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_333", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_334");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_334", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_335");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_335", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_339");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_339", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_340");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_340", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_341");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_341", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_345");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_345", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_346");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_346", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_347");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_347", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_351");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_351", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_352");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_352", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_353");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_353", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_357");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_357", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_358");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_358", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_359");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_359", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_363");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_363", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_364");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_364", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_365");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_365", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_369");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_369", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_370");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_370", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_371");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_371", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_375");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_375", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_376");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_376", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_377");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_377", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_381");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_381", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_382");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_382", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_383");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_383", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_387");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_387", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_388");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_388", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_389");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_389", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_393");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_393", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_394");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_394", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_395");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_395", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_399");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_399", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_400");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_400", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_401");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_401", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_405");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_405", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_406");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_406", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_407");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_407", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_411");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_411", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_412");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_412", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_413");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_413", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_417");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_417", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_418");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_418", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_419");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_419", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_423");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_423", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_424");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_424", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_425");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_425", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_429");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_429", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_430");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_430", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_431");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_431", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_435");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_435", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_436");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_436", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_437");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_437", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_441");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_441", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_442");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_442", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_443");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_443", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_447");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_447", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_448");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_448", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_449");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_449", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_45");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_45", builder.constant({ dataType: "uint4", shape: [32000, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_453");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_453", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_454");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_454", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_455");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_455", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_459");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_459", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_46");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_46", builder.constant({ dataType: "float16", shape: [32000, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_460");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_460", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_461");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_461", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_465");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_465", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_466");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_466", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_467");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_467", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_47");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_47", builder.constant({ dataType: "uint4", shape: [32000, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_471");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_471", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_472");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_472", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_473");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_473", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_477");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_477", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_478");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_478", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_479");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_479", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_483");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_483", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_484");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_484", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_485");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_485", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_489");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_489", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_490");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_490", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_491");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_491", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_495");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_495", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_496");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_496", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_497");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_497", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_501");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_501", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_502");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_502", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_503");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_503", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_507");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_507", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_508");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_508", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_509");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_509", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_51");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_51", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_513");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_513", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_514");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_514", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_515");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_515", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_519");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_519", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_52");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_52", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_520");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_520", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_521");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_521", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_525");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_525", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_526");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_526", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_527");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_527", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_53");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_53", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_531");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_531", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_532");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_532", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_533");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_533", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_537");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_537", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_538");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_538", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_539");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_539", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_543");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_543", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_544");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_544", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_545");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_545", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_549");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_549", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_550");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_550", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_551");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_551", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_555");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_555", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_556");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_556", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_557");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_557", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_561");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_561", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_562");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_562", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_563");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_563", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_567");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_567", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_568");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_568", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_569");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_569", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_57");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_57", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_573");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_573", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_574");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_574", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_575");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_575", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_579");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_579", builder.constant({ dataType: "float16", shape: [32000, 2048] }, buf));
  }
  {
    const sl = weights.getSlice("_58");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_58", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_582");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_582", builder.constant({ dataType: "float16", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_585");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_585", builder.constant({ dataType: "float16", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_589");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_589", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_59");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_59", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_595");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_595", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    // const sl = weights.getSlice("_598");
    // const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    // env.set("_598", builder.constant({ dataType: "uint8", shape: [1] }, buf));

    // This tensor is "webnn_GQA_condition_constant_for_where_1" of shape [1]. It controls when qkv_sequence_length > 1, the key and value are scattered to the
    // beginning of kv cache.
    // See WebNN EP gqa_op_builder.cc for details: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/gqa_op_builder.cc#L356
    // Change it to input for setting prefill and decoding modes.
    env.set("_598", builder.input("webnn_GQA_condition_constant_for_where_1", {dataType: "uint8", shape: [1] }));
  }
  {
    const sl = weights.getSlice("_599");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_599", builder.constant({ dataType: "int32", shape: [1] }, buf));
  }
  {
    // const sl = weights.getSlice("_601");
    // const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    // env.set("_601", builder.constant({ dataType: "int32", shape: [past_sequence_length, 1] }, buf));

    // This tensor is "webnn_GQA_right_constant_of_scatter_indices" of shape [batch_size * qkv_sequence_length * kv_num_heads, 1].
    // See WebNN EP gqa_op_builder.cc for details: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/gqa_op_builder.cc#L347
    // Change it to input to allow dynamic sequence length.
    env.set("_601", builder.input("webnn_GQA_right_constant_of_scatter_indices", {dataType: "int32", shape: [sequence_length, 4, 1] }));
  }
  {
    // const sl = weights.getSlice("_603");
    // const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    // env.set("_603", builder.constant({ dataType: "int32", shape: [past_sequence_length, 2] }, buf));

    // This tensor is "webnn_GQA_left_constant_of_scatter_indices" of shape [batch_size * qkv_sequence_length * kv_num_heads, 2].
    // See WebNN EP gqa_op_builder.cc for details: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/gqa_op_builder.cc#L341
    // Change it to input to allow dynamic sequence length.
    env.set("_603", builder.input("webnn_GQA_left_constant_of_scatter_indices", {dataType: "int32", shape: [sequence_length, 4, 2] }));
  }
  {
    // const sl = weights.getSlice("_610");
    // const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    // env.set("_610", builder.constant({ dataType: "int32", shape: [sequence_length] }, buf));

    // This tensor is "webnn_GQA_pre_neq_right_data_range" of shape [qkv_sequence_length] and then is expanded to shape [past_sequence_length, qkv_sequence_length].
    // See WebNN EP gqa_op_builder.cc for details: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/gqa_op_builder.cc#L491
    // Change it to input to allow dynamic sequence length.
    env.set("_610", builder.input("webnn_GQA_pre_neq_right_data_range", {dataType: "int32", shape: [sequence_length] }));
  }
  {
    const sl = weights.getSlice("_614");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_614", builder.constant({ dataType: "int32", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_618");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_618", builder.constant({ dataType: "float16", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_619");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_619", builder.constant({ dataType: "float16", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_621");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_621", builder.constant({ dataType: "float16", shape: [2048, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_625");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_625", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_626");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_626", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_627");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_627", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_63");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_63", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_638");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_638", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_64");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_64", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_641");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_641", builder.constant({ dataType: "float16", shape: [2048, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_65");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_65", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_656");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_656", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_657");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_657", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_658");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_658", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_669");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_669", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_681");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_681", builder.constant({ dataType: "float16", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_69");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_69", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_695");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_695", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_698");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_698", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_699");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_699", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_70");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_70", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_700");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_700", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_71");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_71", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_715");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_715", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_736");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_736", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_737");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_737", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_738");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_738", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_749");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_749", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_75");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_75", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_76");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_76", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_766");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_766", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_767");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_767", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_768");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_768", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_77");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_77", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_779");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_779", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_804");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_804", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_807");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_807", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_808");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_808", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_809");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_809", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_81");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_81", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_82");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_82", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_824");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_824", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_83");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_83", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_845");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_845", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_846");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_846", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_847");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_847", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_858");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_858", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_87");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_87", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_875");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_875", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_876");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_876", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_877");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_877", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_88");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_88", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_888");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_888", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_89");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_89", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_913");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_913", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_916");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_916", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_917");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_917", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_918");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_918", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_93");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_93", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_933");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_933", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_94");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_94", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_95");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_95", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_954");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_954", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_955");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_955", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_956");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_956", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_967");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_967", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_984");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_984", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_985");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_985", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_986");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_986", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_99");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_99", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_997");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_997", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }

  env.set("_2990", builder.dequantizeLinear(env.get("_2987"), env.get("_2988"), env.get("_2989"), {"axis":2,"blockSize":32,"label":"/model/layers.21/mlp/gate_proj/MatMul_Q4_dequantizeLinear_2617"}));
  env.set("_2991", builder.reshape(env.get("_2990"), [5632,2048]));
  env.set("_2992", builder["transpose"](env.get("_2991"), {"label":"/model/layers.21/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_2619","permutation":[1,0]}));
  env.set("Inserted_1404", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2593","maxValue":2047,"minValue":-2048}));
  env.set("_2962", builder["gather"](env.get("_641"), env.get("Inserted_1404"), {"axis":0,"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_gather_cos_2592"}));
  env.set("_2963", builder.reshape(env.get("_2962"), [1,sequence_length,1,1,32]));
  env.set("_2949", builder.dequantizeLinear(env.get("_2946"), env.get("_2947"), env.get("_2948"), {"axis":2,"blockSize":32,"label":"/model/layers.21/attn/q_proj/MatMul_Q4_dequantizeLinear_2581"}));
  env.set("_2950", builder.reshape(env.get("_2949"), [2048,2048]));
  env.set("_2951", builder["transpose"](env.get("_2950"), {"label":"/model/layers.21/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_2583","permutation":[1,0]}));
  env.set("_2881", builder.dequantizeLinear(env.get("_2878"), env.get("_2879"), env.get("_2880"), {"axis":2,"blockSize":32,"label":"/model/layers.20/mlp/gate_proj/MatMul_Q4_dequantizeLinear_2510"}));
  env.set("_2882", builder.reshape(env.get("_2881"), [5632,2048]));
  env.set("_2883", builder["transpose"](env.get("_2882"), {"label":"/model/layers.20/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_2512","permutation":[1,0]}));
  env.set("Inserted_1344", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2486","maxValue":2047,"minValue":-2048}));
  env.set("_2853", builder["gather"](env.get("_641"), env.get("Inserted_1344"), {"axis":0,"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_gather_cos_2485"}));
  env.set("_2854", builder.reshape(env.get("_2853"), [1,sequence_length,1,1,32]));
  env.set("_2840", builder.dequantizeLinear(env.get("_2837"), env.get("_2838"), env.get("_2839"), {"axis":2,"blockSize":32,"label":"/model/layers.20/attn/q_proj/MatMul_Q4_dequantizeLinear_2474"}));
  env.set("_2841", builder.reshape(env.get("_2840"), [2048,2048]));
  env.set("_2842", builder["transpose"](env.get("_2841"), {"label":"/model/layers.20/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_2476","permutation":[1,0]}));
  env.set("_2772", builder.dequantizeLinear(env.get("_2769"), env.get("_2770"), env.get("_2771"), {"axis":2,"blockSize":32,"label":"/model/layers.19/mlp/gate_proj/MatMul_Q4_dequantizeLinear_2403"}));
  env.set("_2773", builder.reshape(env.get("_2772"), [5632,2048]));
  env.set("_2774", builder["transpose"](env.get("_2773"), {"label":"/model/layers.19/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_2405","permutation":[1,0]}));
  env.set("Inserted_1284", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2379","maxValue":2047,"minValue":-2048}));
  env.set("_2744", builder["gather"](env.get("_641"), env.get("Inserted_1284"), {"axis":0,"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_gather_cos_2378"}));
  env.set("_2745", builder.reshape(env.get("_2744"), [1,sequence_length,1,1,32]));
  env.set("_2731", builder.dequantizeLinear(env.get("_2728"), env.get("_2729"), env.get("_2730"), {"axis":2,"blockSize":32,"label":"/model/layers.19/attn/q_proj/MatMul_Q4_dequantizeLinear_2367"}));
  env.set("_2732", builder.reshape(env.get("_2731"), [2048,2048]));
  env.set("_2733", builder["transpose"](env.get("_2732"), {"label":"/model/layers.19/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_2369","permutation":[1,0]}));
  env.set("_2663", builder.dequantizeLinear(env.get("_2660"), env.get("_2661"), env.get("_2662"), {"axis":2,"blockSize":32,"label":"/model/layers.18/mlp/gate_proj/MatMul_Q4_dequantizeLinear_2296"}));
  env.set("_2664", builder.reshape(env.get("_2663"), [5632,2048]));
  env.set("_2665", builder["transpose"](env.get("_2664"), {"label":"/model/layers.18/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_2298","permutation":[1,0]}));
  env.set("Inserted_1224", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2272","maxValue":2047,"minValue":-2048}));
  env.set("_2635", builder["gather"](env.get("_641"), env.get("Inserted_1224"), {"axis":0,"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_gather_cos_2271"}));
  env.set("_2636", builder.reshape(env.get("_2635"), [1,sequence_length,1,1,32]));
  env.set("_2622", builder.dequantizeLinear(env.get("_2619"), env.get("_2620"), env.get("_2621"), {"axis":2,"blockSize":32,"label":"/model/layers.18/attn/q_proj/MatMul_Q4_dequantizeLinear_2260"}));
  env.set("_2623", builder.reshape(env.get("_2622"), [2048,2048]));
  env.set("_2624", builder["transpose"](env.get("_2623"), {"label":"/model/layers.18/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_2262","permutation":[1,0]}));
  env.set("_2554", builder.dequantizeLinear(env.get("_2551"), env.get("_2552"), env.get("_2553"), {"axis":2,"blockSize":32,"label":"/model/layers.17/mlp/gate_proj/MatMul_Q4_dequantizeLinear_2189"}));
  env.set("_2555", builder.reshape(env.get("_2554"), [5632,2048]));
  env.set("_2556", builder["transpose"](env.get("_2555"), {"label":"/model/layers.17/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_2191","permutation":[1,0]}));
  env.set("Inserted_1164", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2165","maxValue":2047,"minValue":-2048}));
  env.set("_2526", builder["gather"](env.get("_641"), env.get("Inserted_1164"), {"axis":0,"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_gather_cos_2164"}));
  env.set("_2527", builder.reshape(env.get("_2526"), [1,sequence_length,1,1,32]));
  env.set("_2513", builder.dequantizeLinear(env.get("_2510"), env.get("_2511"), env.get("_2512"), {"axis":2,"blockSize":32,"label":"/model/layers.17/attn/q_proj/MatMul_Q4_dequantizeLinear_2153"}));
  env.set("_2514", builder.reshape(env.get("_2513"), [2048,2048]));
  env.set("_2515", builder["transpose"](env.get("_2514"), {"label":"/model/layers.17/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_2155","permutation":[1,0]}));
  env.set("_2445", builder.dequantizeLinear(env.get("_2442"), env.get("_2443"), env.get("_2444"), {"axis":2,"blockSize":32,"label":"/model/layers.16/mlp/gate_proj/MatMul_Q4_dequantizeLinear_2082"}));
  env.set("_2446", builder.reshape(env.get("_2445"), [5632,2048]));
  env.set("_2447", builder["transpose"](env.get("_2446"), {"label":"/model/layers.16/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_2084","permutation":[1,0]}));
  env.set("Inserted_1104", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2058","maxValue":2047,"minValue":-2048}));
  env.set("_2417", builder["gather"](env.get("_641"), env.get("Inserted_1104"), {"axis":0,"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_gather_cos_2057"}));
  env.set("_2418", builder.reshape(env.get("_2417"), [1,sequence_length,1,1,32]));
  env.set("_2404", builder.dequantizeLinear(env.get("_2401"), env.get("_2402"), env.get("_2403"), {"axis":2,"blockSize":32,"label":"/model/layers.16/attn/q_proj/MatMul_Q4_dequantizeLinear_2046"}));
  env.set("_2405", builder.reshape(env.get("_2404"), [2048,2048]));
  env.set("_2406", builder["transpose"](env.get("_2405"), {"label":"/model/layers.16/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_2048","permutation":[1,0]}));
  env.set("_2336", builder.dequantizeLinear(env.get("_2333"), env.get("_2334"), env.get("_2335"), {"axis":2,"blockSize":32,"label":"/model/layers.15/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1975"}));
  env.set("_2337", builder.reshape(env.get("_2336"), [5632,2048]));
  env.set("_2338", builder["transpose"](env.get("_2337"), {"label":"/model/layers.15/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1977","permutation":[1,0]}));
  env.set("Inserted_1044", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1951","maxValue":2047,"minValue":-2048}));
  env.set("_2308", builder["gather"](env.get("_641"), env.get("Inserted_1044"), {"axis":0,"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_gather_cos_1950"}));
  env.set("_2309", builder.reshape(env.get("_2308"), [1,sequence_length,1,1,32]));
  env.set("_2295", builder.dequantizeLinear(env.get("_2292"), env.get("_2293"), env.get("_2294"), {"axis":2,"blockSize":32,"label":"/model/layers.15/attn/q_proj/MatMul_Q4_dequantizeLinear_1939"}));
  env.set("_2296", builder.reshape(env.get("_2295"), [2048,2048]));
  env.set("_2297", builder["transpose"](env.get("_2296"), {"label":"/model/layers.15/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1941","permutation":[1,0]}));
  env.set("_2227", builder.dequantizeLinear(env.get("_2224"), env.get("_2225"), env.get("_2226"), {"axis":2,"blockSize":32,"label":"/model/layers.14/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1868"}));
  env.set("_2228", builder.reshape(env.get("_2227"), [5632,2048]));
  env.set("_2229", builder["transpose"](env.get("_2228"), {"label":"/model/layers.14/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1870","permutation":[1,0]}));
  env.set("Inserted_984", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1844","maxValue":2047,"minValue":-2048}));
  env.set("_2199", builder["gather"](env.get("_641"), env.get("Inserted_984"), {"axis":0,"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_gather_cos_1843"}));
  env.set("_2200", builder.reshape(env.get("_2199"), [1,sequence_length,1,1,32]));
  env.set("_2186", builder.dequantizeLinear(env.get("_2183"), env.get("_2184"), env.get("_2185"), {"axis":2,"blockSize":32,"label":"/model/layers.14/attn/q_proj/MatMul_Q4_dequantizeLinear_1832"}));
  env.set("_2187", builder.reshape(env.get("_2186"), [2048,2048]));
  env.set("_2188", builder["transpose"](env.get("_2187"), {"label":"/model/layers.14/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1834","permutation":[1,0]}));
  env.set("_2118", builder.dequantizeLinear(env.get("_2115"), env.get("_2116"), env.get("_2117"), {"axis":2,"blockSize":32,"label":"/model/layers.13/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1761"}));
  env.set("_2119", builder.reshape(env.get("_2118"), [5632,2048]));
  env.set("_2120", builder["transpose"](env.get("_2119"), {"label":"/model/layers.13/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1763","permutation":[1,0]}));
  env.set("Inserted_924", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1737","maxValue":2047,"minValue":-2048}));
  env.set("_2090", builder["gather"](env.get("_641"), env.get("Inserted_924"), {"axis":0,"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_gather_cos_1736"}));
  env.set("_2091", builder.reshape(env.get("_2090"), [1,sequence_length,1,1,32]));
  env.set("_2077", builder.dequantizeLinear(env.get("_2074"), env.get("_2075"), env.get("_2076"), {"axis":2,"blockSize":32,"label":"/model/layers.13/attn/q_proj/MatMul_Q4_dequantizeLinear_1725"}));
  env.set("_2078", builder.reshape(env.get("_2077"), [2048,2048]));
  env.set("_2079", builder["transpose"](env.get("_2078"), {"label":"/model/layers.13/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1727","permutation":[1,0]}));
  env.set("_2009", builder.dequantizeLinear(env.get("_2006"), env.get("_2007"), env.get("_2008"), {"axis":2,"blockSize":32,"label":"/model/layers.12/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1654"}));
  env.set("_2010", builder.reshape(env.get("_2009"), [5632,2048]));
  env.set("_2011", builder["transpose"](env.get("_2010"), {"label":"/model/layers.12/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1656","permutation":[1,0]}));
  env.set("Inserted_864", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1630","maxValue":2047,"minValue":-2048}));
  env.set("_1981", builder["gather"](env.get("_641"), env.get("Inserted_864"), {"axis":0,"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_gather_cos_1629"}));
  env.set("_1982", builder.reshape(env.get("_1981"), [1,sequence_length,1,1,32]));
  env.set("_1968", builder.dequantizeLinear(env.get("_1965"), env.get("_1966"), env.get("_1967"), {"axis":2,"blockSize":32,"label":"/model/layers.12/attn/q_proj/MatMul_Q4_dequantizeLinear_1618"}));
  env.set("_1969", builder.reshape(env.get("_1968"), [2048,2048]));
  env.set("_1970", builder["transpose"](env.get("_1969"), {"label":"/model/layers.12/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1620","permutation":[1,0]}));
  env.set("_1900", builder.dequantizeLinear(env.get("_1897"), env.get("_1898"), env.get("_1899"), {"axis":2,"blockSize":32,"label":"/model/layers.11/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1547"}));
  env.set("_1901", builder.reshape(env.get("_1900"), [5632,2048]));
  env.set("_1902", builder["transpose"](env.get("_1901"), {"label":"/model/layers.11/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1549","permutation":[1,0]}));
  env.set("Inserted_804", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1523","maxValue":2047,"minValue":-2048}));
  env.set("_1872", builder["gather"](env.get("_641"), env.get("Inserted_804"), {"axis":0,"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_gather_cos_1522"}));
  env.set("_1873", builder.reshape(env.get("_1872"), [1,sequence_length,1,1,32]));
  env.set("_1859", builder.dequantizeLinear(env.get("_1856"), env.get("_1857"), env.get("_1858"), {"axis":2,"blockSize":32,"label":"/model/layers.11/attn/q_proj/MatMul_Q4_dequantizeLinear_1511"}));
  env.set("_1860", builder.reshape(env.get("_1859"), [2048,2048]));
  env.set("_1861", builder["transpose"](env.get("_1860"), {"label":"/model/layers.11/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1513","permutation":[1,0]}));
  env.set("_1791", builder.dequantizeLinear(env.get("_1788"), env.get("_1789"), env.get("_1790"), {"axis":2,"blockSize":32,"label":"/model/layers.10/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1440"}));
  env.set("_1792", builder.reshape(env.get("_1791"), [5632,2048]));
  env.set("_1793", builder["transpose"](env.get("_1792"), {"label":"/model/layers.10/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1442","permutation":[1,0]}));
  env.set("Inserted_744", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1416","maxValue":2047,"minValue":-2048}));
  env.set("_1763", builder["gather"](env.get("_641"), env.get("Inserted_744"), {"axis":0,"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_gather_cos_1415"}));
  env.set("_1764", builder.reshape(env.get("_1763"), [1,sequence_length,1,1,32]));
  env.set("_1750", builder.dequantizeLinear(env.get("_1747"), env.get("_1748"), env.get("_1749"), {"axis":2,"blockSize":32,"label":"/model/layers.10/attn/q_proj/MatMul_Q4_dequantizeLinear_1404"}));
  env.set("_1751", builder.reshape(env.get("_1750"), [2048,2048]));
  env.set("_1752", builder["transpose"](env.get("_1751"), {"label":"/model/layers.10/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1406","permutation":[1,0]}));
  env.set("_1682", builder.dequantizeLinear(env.get("_1679"), env.get("_1680"), env.get("_1681"), {"axis":2,"blockSize":32,"label":"/model/layers.9/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1333"}));
  env.set("_1683", builder.reshape(env.get("_1682"), [5632,2048]));
  env.set("_1684", builder["transpose"](env.get("_1683"), {"label":"/model/layers.9/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1335","permutation":[1,0]}));
  env.set("Inserted_684", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1309","maxValue":2047,"minValue":-2048}));
  env.set("_1654", builder["gather"](env.get("_641"), env.get("Inserted_684"), {"axis":0,"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_gather_cos_1308"}));
  env.set("_1655", builder.reshape(env.get("_1654"), [1,sequence_length,1,1,32]));
  env.set("_1641", builder.dequantizeLinear(env.get("_1638"), env.get("_1639"), env.get("_1640"), {"axis":2,"blockSize":32,"label":"/model/layers.9/attn/q_proj/MatMul_Q4_dequantizeLinear_1297"}));
  env.set("_1642", builder.reshape(env.get("_1641"), [2048,2048]));
  env.set("_1643", builder["transpose"](env.get("_1642"), {"label":"/model/layers.9/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1299","permutation":[1,0]}));
  env.set("_1573", builder.dequantizeLinear(env.get("_1570"), env.get("_1571"), env.get("_1572"), {"axis":2,"blockSize":32,"label":"/model/layers.8/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1226"}));
  env.set("_1574", builder.reshape(env.get("_1573"), [5632,2048]));
  env.set("_1575", builder["transpose"](env.get("_1574"), {"label":"/model/layers.8/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1228","permutation":[1,0]}));
  env.set("Inserted_624", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1202","maxValue":2047,"minValue":-2048}));
  env.set("_1545", builder["gather"](env.get("_641"), env.get("Inserted_624"), {"axis":0,"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_gather_cos_1201"}));
  env.set("_1546", builder.reshape(env.get("_1545"), [1,sequence_length,1,1,32]));
  env.set("_1532", builder.dequantizeLinear(env.get("_1529"), env.get("_1530"), env.get("_1531"), {"axis":2,"blockSize":32,"label":"/model/layers.8/attn/q_proj/MatMul_Q4_dequantizeLinear_1190"}));
  env.set("_1533", builder.reshape(env.get("_1532"), [2048,2048]));
  env.set("_1534", builder["transpose"](env.get("_1533"), {"label":"/model/layers.8/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1192","permutation":[1,0]}));
  env.set("_1464", builder.dequantizeLinear(env.get("_1461"), env.get("_1462"), env.get("_1463"), {"axis":2,"blockSize":32,"label":"/model/layers.7/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1119"}));
  env.set("_1465", builder.reshape(env.get("_1464"), [5632,2048]));
  env.set("_1466", builder["transpose"](env.get("_1465"), {"label":"/model/layers.7/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1121","permutation":[1,0]}));
  env.set("Inserted_564", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1095","maxValue":2047,"minValue":-2048}));
  env.set("_1436", builder["gather"](env.get("_641"), env.get("Inserted_564"), {"axis":0,"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_gather_cos_1094"}));
  env.set("_1437", builder.reshape(env.get("_1436"), [1,sequence_length,1,1,32]));
  env.set("_1423", builder.dequantizeLinear(env.get("_1420"), env.get("_1421"), env.get("_1422"), {"axis":2,"blockSize":32,"label":"/model/layers.7/attn/q_proj/MatMul_Q4_dequantizeLinear_1083"}));
  env.set("_1424", builder.reshape(env.get("_1423"), [2048,2048]));
  env.set("_1425", builder["transpose"](env.get("_1424"), {"label":"/model/layers.7/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1085","permutation":[1,0]}));
  env.set("_1355", builder.dequantizeLinear(env.get("_1352"), env.get("_1353"), env.get("_1354"), {"axis":2,"blockSize":32,"label":"/model/layers.6/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1012"}));
  env.set("_1356", builder.reshape(env.get("_1355"), [5632,2048]));
  env.set("_1357", builder["transpose"](env.get("_1356"), {"label":"/model/layers.6/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1014","permutation":[1,0]}));
  env.set("Inserted_504", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_988","maxValue":2047,"minValue":-2048}));
  env.set("_1327", builder["gather"](env.get("_641"), env.get("Inserted_504"), {"axis":0,"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_gather_cos_987"}));
  env.set("_1328", builder.reshape(env.get("_1327"), [1,sequence_length,1,1,32]));
  env.set("_1314", builder.dequantizeLinear(env.get("_1311"), env.get("_1312"), env.get("_1313"), {"axis":2,"blockSize":32,"label":"/model/layers.6/attn/q_proj/MatMul_Q4_dequantizeLinear_976"}));
  env.set("_1315", builder.reshape(env.get("_1314"), [2048,2048]));
  env.set("_1316", builder["transpose"](env.get("_1315"), {"label":"/model/layers.6/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_978","permutation":[1,0]}));
  env.set("_1246", builder.dequantizeLinear(env.get("_1243"), env.get("_1244"), env.get("_1245"), {"axis":2,"blockSize":32,"label":"/model/layers.5/mlp/gate_proj/MatMul_Q4_dequantizeLinear_905"}));
  env.set("_1247", builder.reshape(env.get("_1246"), [5632,2048]));
  env.set("_1248", builder["transpose"](env.get("_1247"), {"label":"/model/layers.5/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_907","permutation":[1,0]}));
  env.set("Inserted_444", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_881","maxValue":2047,"minValue":-2048}));
  env.set("_1218", builder["gather"](env.get("_641"), env.get("Inserted_444"), {"axis":0,"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_gather_cos_880"}));
  env.set("_1219", builder.reshape(env.get("_1218"), [1,sequence_length,1,1,32]));
  env.set("_1205", builder.dequantizeLinear(env.get("_1202"), env.get("_1203"), env.get("_1204"), {"axis":2,"blockSize":32,"label":"/model/layers.5/attn/q_proj/MatMul_Q4_dequantizeLinear_869"}));
  env.set("_1206", builder.reshape(env.get("_1205"), [2048,2048]));
  env.set("_1207", builder["transpose"](env.get("_1206"), {"label":"/model/layers.5/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_871","permutation":[1,0]}));
  env.set("_1137", builder.dequantizeLinear(env.get("_1134"), env.get("_1135"), env.get("_1136"), {"axis":2,"blockSize":32,"label":"/model/layers.4/mlp/gate_proj/MatMul_Q4_dequantizeLinear_798"}));
  env.set("_1138", builder.reshape(env.get("_1137"), [5632,2048]));
  env.set("_1139", builder["transpose"](env.get("_1138"), {"label":"/model/layers.4/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_800","permutation":[1,0]}));
  env.set("Inserted_384", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_774","maxValue":2047,"minValue":-2048}));
  env.set("_1109", builder["gather"](env.get("_641"), env.get("Inserted_384"), {"axis":0,"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_gather_cos_773"}));
  env.set("_1110", builder.reshape(env.get("_1109"), [1,sequence_length,1,1,32]));
  env.set("_1096", builder.dequantizeLinear(env.get("_1093"), env.get("_1094"), env.get("_1095"), {"axis":2,"blockSize":32,"label":"/model/layers.4/attn/q_proj/MatMul_Q4_dequantizeLinear_762"}));
  env.set("_1097", builder.reshape(env.get("_1096"), [2048,2048]));
  env.set("_1098", builder["transpose"](env.get("_1097"), {"label":"/model/layers.4/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_764","permutation":[1,0]}));
  env.set("_1028", builder.dequantizeLinear(env.get("_1025"), env.get("_1026"), env.get("_1027"), {"axis":2,"blockSize":32,"label":"/model/layers.3/mlp/gate_proj/MatMul_Q4_dequantizeLinear_691"}));
  env.set("_1029", builder.reshape(env.get("_1028"), [5632,2048]));
  env.set("_1030", builder["transpose"](env.get("_1029"), {"label":"/model/layers.3/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_693","permutation":[1,0]}));
  env.set("Inserted_324", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_667","maxValue":2047,"minValue":-2048}));
  env.set("_1000", builder["gather"](env.get("_641"), env.get("Inserted_324"), {"axis":0,"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_gather_cos_666"}));
  env.set("_1001", builder.reshape(env.get("_1000"), [1,sequence_length,1,1,32]));
  env.set("_987", builder.dequantizeLinear(env.get("_984"), env.get("_985"), env.get("_986"), {"axis":2,"blockSize":32,"label":"/model/layers.3/attn/q_proj/MatMul_Q4_dequantizeLinear_655"}));
  env.set("_988", builder.reshape(env.get("_987"), [2048,2048]));
  env.set("_989", builder["transpose"](env.get("_988"), {"label":"/model/layers.3/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_657","permutation":[1,0]}));
  env.set("_919", builder.dequantizeLinear(env.get("_916"), env.get("_917"), env.get("_918"), {"axis":2,"blockSize":32,"label":"/model/layers.2/mlp/gate_proj/MatMul_Q4_dequantizeLinear_584"}));
  env.set("_920", builder.reshape(env.get("_919"), [5632,2048]));
  env.set("_921", builder["transpose"](env.get("_920"), {"label":"/model/layers.2/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_586","permutation":[1,0]}));
  env.set("Inserted_264", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_560","maxValue":2047,"minValue":-2048}));
  env.set("_891", builder["gather"](env.get("_641"), env.get("Inserted_264"), {"axis":0,"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_gather_cos_559"}));
  env.set("_892", builder.reshape(env.get("_891"), [1,sequence_length,1,1,32]));
  env.set("_878", builder.dequantizeLinear(env.get("_875"), env.get("_876"), env.get("_877"), {"axis":2,"blockSize":32,"label":"/model/layers.2/attn/q_proj/MatMul_Q4_dequantizeLinear_548"}));
  env.set("_879", builder.reshape(env.get("_878"), [2048,2048]));
  env.set("_880", builder["transpose"](env.get("_879"), {"label":"/model/layers.2/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_550","permutation":[1,0]}));
  env.set("_810", builder.dequantizeLinear(env.get("_807"), env.get("_808"), env.get("_809"), {"axis":2,"blockSize":32,"label":"/model/layers.1/mlp/gate_proj/MatMul_Q4_dequantizeLinear_477"}));
  env.set("_811", builder.reshape(env.get("_810"), [5632,2048]));
  env.set("_812", builder["transpose"](env.get("_811"), {"label":"/model/layers.1/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_479","permutation":[1,0]}));
  env.set("Inserted_204", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_453","maxValue":2047,"minValue":-2048}));
  env.set("_782", builder["gather"](env.get("_641"), env.get("Inserted_204"), {"axis":0,"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_gather_cos_452"}));
  env.set("_783", builder.reshape(env.get("_782"), [1,sequence_length,1,1,32]));
  env.set("_769", builder.dequantizeLinear(env.get("_766"), env.get("_767"), env.get("_768"), {"axis":2,"blockSize":32,"label":"/model/layers.1/attn/q_proj/MatMul_Q4_dequantizeLinear_441"}));
  env.set("_770", builder.reshape(env.get("_769"), [2048,2048]));
  env.set("_771", builder["transpose"](env.get("_770"), {"label":"/model/layers.1/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_443","permutation":[1,0]}));
  env.set("_701", builder.dequantizeLinear(env.get("_698"), env.get("_699"), env.get("_700"), {"axis":2,"blockSize":32,"label":"/model/layers.0/mlp/gate_proj/MatMul_Q4_dequantizeLinear_370"}));
  env.set("_702", builder.reshape(env.get("_701"), [5632,2048]));
  env.set("_703", builder["transpose"](env.get("_702"), {"label":"/model/layers.0/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_372","permutation":[1,0]}));
  env.set("Inserted_144", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_346","maxValue":2047,"minValue":-2048}));
  env.set("_672", builder["gather"](env.get("_641"), env.get("Inserted_144"), {"axis":0,"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_gather_cos_345"}));
  env.set("_673", builder.reshape(env.get("_672"), [1,sequence_length,1,1,32]));
  env.set("_659", builder.dequantizeLinear(env.get("_656"), env.get("_657"), env.get("_658"), {"axis":2,"blockSize":32,"label":"/model/layers.0/attn/q_proj/MatMul_Q4_dequantizeLinear_334"}));
  env.set("_660", builder.reshape(env.get("_659"), [2048,2048]));
  env.set("_661", builder["transpose"](env.get("_660"), {"label":"/model/layers.0/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_336","permutation":[1,0]}));
  env.set("Inserted_90", builder["clamp"](env.get("input_ids_580"), {"label":"Inserted_Clip_268","maxValue":31999,"minValue":-32000}));
  env.set("_581", builder["gather"](env.get("_579"), env.get("Inserted_90"), {"axis":0,"label":"/model/embed_tokens/Gather_267"}));
  env.set("_583", builder["pow"](env.get("_581"), env.get("_582"), {"label":"/model/layers.0/input_layernorm/LayerNorm_pow_269"}));
  env.set("_584", builder["reduceMean"](env.get("_583"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.0/input_layernorm/LayerNorm_reduceMean_270"}));
  env.set("_586", builder["add"](env.get("_584"), env.get("_585"), {"label":"/model/layers.0/input_layernorm/LayerNorm_add_271"}));
  env.set("_587", builder["sqrt"](env.get("_586"), {"label":"/model/layers.0/input_layernorm/LayerNorm_sqrt_272"}));
  env.set("_588", builder["div"](env.get("_581"), env.get("_587"), {"label":"/model/layers.0/input_layernorm/LayerNorm_div_273"}));
  env.set("_590", builder["mul"](env.get("_589"), env.get("_588"), {"label":"/model/layers.0/input_layernorm/LayerNorm_mul_274"}));
  env.set("_662", builder["matmul"](env.get("_590"), env.get("_661"), {"label":"/model/layers.0/attn/q_proj/MatMul_Q4_matmul_337"}));
  env.set("_663", builder.reshape(env.get("_662"), [1,sequence_length,32,64]));
  env.set("_664", builder.reshape(env.get("_663"), [1,sequence_length,32,2,32]));
  env.set("_674", builder["mul"](env.get("_664"), env.get("_673"), {"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_mul_cos_348"}));
  env.set("_675", builder.reshape(env.get("_674"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_664"), 2, {"axis":3,"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_split_partial_input0_340"});
    env.set("_665", tmp[0]);
    env.set("_666", tmp[1]);
  }
  env.set("_667", builder.concat([env.get("_666"), env.get("_665")], 3, {"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_concat_partial_input0_341"}));
  env.set("Inserted_135", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_332","maxValue":2047,"minValue":-2048}));
  env.set("_654", builder["gather"](env.get("_621"), env.get("Inserted_135"), {"axis":0,"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_gather_sin_331"}));
  env.set("_655", builder.reshape(env.get("_654"), [1,sequence_length,1,1,32]));
  env.set("_668", builder["mul"](env.get("_667"), env.get("_655"), {"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_mul_sin_342"}));
  env.set("_670", builder["mul"](env.get("_668"), env.get("_669"), {"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_mul_sign_343"}));
  env.set("_671", builder.reshape(env.get("_670"), [1,sequence_length,32,64]));
  env.set("_676", builder["add"](env.get("_675"), env.get("_671"), {"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_350"}));
  env.set("_677", builder.reshape(env.get("_676"), [1,sequence_length,2048]));
  env.set("_678", builder.reshape(env.get("_677"), [1,sequence_length,32,64]));
  env.set("_679", builder["transpose"](env.get("_678"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/query/transpose_353","permutation":[0,2,1,3]}));
  env.set("Inserted_96", builder.cast(env.get("_598"), "uint8"));
  env.set("_594", builder["reduceSum"](env.get("attention_mask_593"), {"axes":[1],"keepDimensions":true,"label":"/model/attn_mask_reformat/attn_mask_subgraph/ReduceSum_277"}));
  env.set("_596", builder["sub"](env.get("_594"), env.get("_595"), {"label":"/model/attn_mask_reformat/attn_mask_subgraph/Sub_278"}));
  env.set("_597", builder.cast(env.get("_596"), "int32"));
  env.set("_600", builder["where"](env.get("Inserted_96"), env.get("_599"), env.get("_597"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/scatter/where_280"}));
  env.set("_602", builder["add"](env.get("_601"), env.get("_600"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/right_constant/add_282"}));
  env.set("_604", builder.concat([env.get("_603"), env.get("_602")], 2, {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_283"}));
  env.set("_605", builder.reshape(env.get("_604"), [1,sequence_length,4,3]));
  env.set("Inserted_127", builder.cast(env.get("_605"), "int64"));
  env.set("Inserted_129", builder["max"](env.get("Inserted_127"), env.get("Inserted_128"), {"label":"Inserted_Max_325"}));
  env.set("Inserted_131", builder["min"](env.get("Inserted_129"), env.get("Inserted_130"), {"label":"Inserted_Min_326"}));
  env.set("Inserted_120", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_316","maxValue":2047,"minValue":-2048}));
  env.set("_642", builder["gather"](env.get("_641"), env.get("Inserted_120"), {"axis":0,"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_gather_cos_315"}));
  env.set("_643", builder.reshape(env.get("_642"), [1,sequence_length,1,1,32]));
  env.set("_628", builder.dequantizeLinear(env.get("_625"), env.get("_626"), env.get("_627"), {"axis":2,"blockSize":32,"label":"/model/layers.0/attn/k_proj/MatMul_Q4_dequantizeLinear_304"}));
  env.set("_629", builder.reshape(env.get("_628"), [256,2048]));
  env.set("_630", builder["transpose"](env.get("_629"), {"label":"/model/layers.0/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_306","permutation":[1,0]}));
  env.set("_631", builder["matmul"](env.get("_590"), env.get("_630"), {"label":"/model/layers.0/attn/k_proj/MatMul_Q4_matmul_307"}));
  env.set("_632", builder.reshape(env.get("_631"), [1,sequence_length,4,64]));
  env.set("_633", builder.reshape(env.get("_632"), [1,sequence_length,4,2,32]));
  env.set("_644", builder["mul"](env.get("_633"), env.get("_643"), {"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_mul_cos_318"}));
  env.set("_645", builder.reshape(env.get("_644"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_633"), 2, {"axis":3,"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_split_partial_input0_310"});
    env.set("_634", tmp[0]);
    env.set("_635", tmp[1]);
  }
  env.set("_636", builder.concat([env.get("_635"), env.get("_634")], 3, {"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_concat_partial_input0_311"}));
  env.set("Inserted_111", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_302","maxValue":2047,"minValue":-2048}));
  env.set("_623", builder["gather"](env.get("_621"), env.get("Inserted_111"), {"axis":0,"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_gather_sin_301"}));
  env.set("_624", builder.reshape(env.get("_623"), [1,sequence_length,1,1,32]));
  env.set("_637", builder["mul"](env.get("_636"), env.get("_624"), {"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_mul_sin_312"}));
  env.set("_639", builder["mul"](env.get("_637"), env.get("_638"), {"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_mul_sign_313"}));
  env.set("_640", builder.reshape(env.get("_639"), [1,sequence_length,4,64]));
  env.set("_646", builder["add"](env.get("_645"), env.get("_640"), {"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_320"}));
  env.set("_647", builder.reshape(env.get("_646"), [1,sequence_length,256]));
  env.set("_648", builder.reshape(env.get("_647"), [1,sequence_length,4,64]));
  env.set("present_0_key_0", builder["scatterND"](env.get("past_key_values_0_key_649"), env.get("Inserted_131"), env.get("_648"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/present_key/ScatterND_323"}));
  env.set("_650", builder.reshape(env.get("present_0_key_0"), [1,4,1,past_sequence_length,64]));
  env.set("_651", builder.expand(env.get("_650"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/true_present_key/expand_328"}));
  env.set("_652", builder.reshape(env.get("_651"), [1,32,past_sequence_length,64]));
  env.set("_653", builder["transpose"](env.get("_652"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/present_key/transpose_330","permutation":[0,1,3,2]}));
  env.set("_680", builder["matmul"](env.get("_679"), env.get("_653"), {"label":"/model/layers.0/attn/GroupQueryAttention_/Attention/qkv/matmul_1_354"}));
  env.set("_682", builder["mul"](env.get("_680"), env.get("_681"), {"label":"/model/layers.0/attn/GroupQueryAttention_/Attention/qkv/div_355"}));
  env.set("_615", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_295"}));
  env.set("_616", builder.cumulativeSum(env.get("_615"), 3, {"exclusive":true,"label":"/model/layers.0/attn/GroupQueryAttention_range_of_mask_shape_296"}));
  env.set("_611", builder["add"](env.get("_610"), env.get("_600"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/attn_mask/add_292"}));
  env.set("_612", builder.expand(env.get("_611"), [past_sequence_length,sequence_length], {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/expand_neq_right_293"}));
  env.set("_613", builder["transpose"](env.get("_612"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/neq_right/transpose_294","permutation":[1,0]}));
  env.set("Inserted_109", builder["lesser"](env.get("_616"), env.get("_613"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_297"}));
  env.set("_617", builder.cast(env.get("Inserted_109"), "uint8"));
  env.set("Inserted_110", builder.cast(env.get("_617"), "uint8"));
  env.set("_620", builder["where"](env.get("Inserted_110"), env.get("_618"), env.get("_619"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/attn_mask/where_299"}));
  env.set("_683", builder["add"](env.get("_682"), env.get("_620"), {"label":"/model/layers.0/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_356"}));
  env.set("_684", builder["softmax"](env.get("_683"), 3));
  env.set("Inserted_98", builder.cast(env.get("_605"), "int64"));
  env.set("Inserted_100", builder["max"](env.get("Inserted_98"), env.get("Inserted_99"), {"label":"Inserted_Max_287"}));
  env.set("Inserted_102", builder["min"](env.get("Inserted_100"), env.get("Inserted_101"), {"label":"Inserted_Min_288"}));
  env.set("_576", builder.dequantizeLinear(env.get("_573"), env.get("_574"), env.get("_575"), {"axis":2,"blockSize":32,"label":"/model/layers.0/attn/v_proj/MatMul_Q4_dequantizeLinear_264"}));
  env.set("_577", builder.reshape(env.get("_576"), [256,2048]));
  env.set("_578", builder["transpose"](env.get("_577"), {"label":"/model/layers.0/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_266","permutation":[1,0]}));
  env.set("_591", builder["matmul"](env.get("_590"), env.get("_578"), {"label":"/model/layers.0/attn/v_proj/MatMul_Q4_matmul_275"}));
  env.set("_592", builder.reshape(env.get("_591"), [1,sequence_length,4,64]));
  env.set("present_0_value_1", builder["scatterND"](env.get("past_key_values_0_value_606"), env.get("Inserted_102"), env.get("_592"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/present_value/ScatterND_285"}));
  env.set("_607", builder.reshape(env.get("present_0_value_1"), [1,4,1,past_sequence_length,64]));
  env.set("_608", builder.expand(env.get("_607"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/true_present_value/expand_290"}));
  env.set("_609", builder.reshape(env.get("_608"), [1,32,past_sequence_length,64]));
  env.set("_685", builder["matmul"](env.get("_684"), env.get("_609"), {"label":"/model/layers.0/attn/GroupQueryAttention_/Attention/qkv/matmul_2_358"}));
  env.set("_686", builder["transpose"](env.get("_685"), {"label":"/model/layers.0/attn/GroupQueryAttention_/Attention/qkv/transpose_359","permutation":[0,2,1,3]}));
  env.set("_687", builder.reshape(env.get("_686"), [1,sequence_length,2048]));
  env.set("_570", builder.dequantizeLinear(env.get("_567"), env.get("_568"), env.get("_569"), {"axis":2,"blockSize":32,"label":"/model/layers.0/attn/o_proj/MatMul_Q4_dequantizeLinear_261"}));
  env.set("_571", builder.reshape(env.get("_570"), [2048,2048]));
  env.set("_572", builder["transpose"](env.get("_571"), {"label":"/model/layers.0/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_263","permutation":[1,0]}));
  env.set("_688", builder["matmul"](env.get("_687"), env.get("_572"), {"label":"/model/layers.0/attn/o_proj/MatMul_Q4_matmul_361"}));
  env.set("_689", builder["add"](env.get("_581"), env.get("_688"), {"label":"/model/layers.0/post_attention_layernorm/SkipLayerNorm_add_skip_362"}));
  env.set("_690", builder["pow"](env.get("_689"), env.get("_582"), {"label":"/model/layers.0/post_attention_layernorm/SkipLayerNorm_pow_363"}));
  env.set("_691", builder["reduceMean"](env.get("_690"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.0/post_attention_layernorm/SkipLayerNorm_reduceMean_364"}));
  env.set("_692", builder["add"](env.get("_691"), env.get("_585"), {"label":"/model/layers.0/post_attention_layernorm/SkipLayerNorm_add_365"}));
  env.set("_693", builder["sqrt"](env.get("_692"), {"label":"/model/layers.0/post_attention_layernorm/SkipLayerNorm_sqrt_366"}));
  env.set("_694", builder["div"](env.get("_689"), env.get("_693"), {"label":"/model/layers.0/post_attention_layernorm/SkipLayerNorm_div_367"}));
  env.set("_696", builder["mul"](env.get("_695"), env.get("_694"), {"label":"/model/layers.0/post_attention_layernorm/SkipLayerNorm_mul_368"}));
  env.set("_704", builder["matmul"](env.get("_696"), env.get("_703"), {"label":"/model/layers.0/mlp/gate_proj/MatMul_Q4_matmul_373"}));
  env.set("_705", builder["sigmoid"](env.get("_704"), {"label":"/model/layers.0/mlp/act_fn/Sigmoid_374"}));
  env.set("_706", builder["mul"](env.get("_704"), env.get("_705"), {"label":"/model/layers.0/mlp/act_fn/Mul_375"}));
  env.set("_564", builder.dequantizeLinear(env.get("_561"), env.get("_562"), env.get("_563"), {"axis":2,"blockSize":32,"label":"/model/layers.0/mlp/up_proj/MatMul_Q4_dequantizeLinear_258"}));
  env.set("_565", builder.reshape(env.get("_564"), [5632,2048]));
  env.set("_566", builder["transpose"](env.get("_565"), {"label":"/model/layers.0/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_260","permutation":[1,0]}));
  env.set("_697", builder["matmul"](env.get("_696"), env.get("_566"), {"label":"/model/layers.0/mlp/up_proj/MatMul_Q4_matmul_369"}));
  env.set("_707", builder["mul"](env.get("_706"), env.get("_697"), {"label":"/model/layers.0/mlp/Mul_376"}));
  env.set("_558", builder.dequantizeLinear(env.get("_555"), env.get("_556"), env.get("_557"), {"axis":2,"blockSize":32,"label":"/model/layers.0/mlp/down_proj/MatMul_Q4_dequantizeLinear_255"}));
  env.set("_559", builder.reshape(env.get("_558"), [2048,5632]));
  env.set("_560", builder["transpose"](env.get("_559"), {"label":"/model/layers.0/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_257","permutation":[1,0]}));
  env.set("_708", builder["matmul"](env.get("_707"), env.get("_560"), {"label":"/model/layers.0/mlp/down_proj/MatMul_Q4_matmul_377"}));
  env.set("_709", builder["add"](env.get("_689"), env.get("_708"), {"label":"/model/layers.1/input_layernorm/SkipLayerNorm_add_skip_378"}));
  env.set("_710", builder["pow"](env.get("_709"), env.get("_582"), {"label":"/model/layers.1/input_layernorm/SkipLayerNorm_pow_379"}));
  env.set("_711", builder["reduceMean"](env.get("_710"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.1/input_layernorm/SkipLayerNorm_reduceMean_380"}));
  env.set("_712", builder["add"](env.get("_711"), env.get("_585"), {"label":"/model/layers.1/input_layernorm/SkipLayerNorm_add_381"}));
  env.set("_713", builder["sqrt"](env.get("_712"), {"label":"/model/layers.1/input_layernorm/SkipLayerNorm_sqrt_382"}));
  env.set("_714", builder["div"](env.get("_709"), env.get("_713"), {"label":"/model/layers.1/input_layernorm/SkipLayerNorm_div_383"}));
  env.set("_716", builder["mul"](env.get("_715"), env.get("_714"), {"label":"/model/layers.1/input_layernorm/SkipLayerNorm_mul_384"}));
  env.set("_772", builder["matmul"](env.get("_716"), env.get("_771"), {"label":"/model/layers.1/attn/q_proj/MatMul_Q4_matmul_444"}));
  env.set("_773", builder.reshape(env.get("_772"), [1,sequence_length,32,64]));
  env.set("_774", builder.reshape(env.get("_773"), [1,sequence_length,32,2,32]));
  env.set("_784", builder["mul"](env.get("_774"), env.get("_783"), {"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_mul_cos_455"}));
  env.set("_785", builder.reshape(env.get("_784"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_774"), 2, {"axis":3,"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_split_partial_input0_447"});
    env.set("_775", tmp[0]);
    env.set("_776", tmp[1]);
  }
  env.set("_777", builder.concat([env.get("_776"), env.get("_775")], 3, {"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_concat_partial_input0_448"}));
  env.set("Inserted_195", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_439","maxValue":2047,"minValue":-2048}));
  env.set("_764", builder["gather"](env.get("_621"), env.get("Inserted_195"), {"axis":0,"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_gather_sin_438"}));
  env.set("_765", builder.reshape(env.get("_764"), [1,sequence_length,1,1,32]));
  env.set("_778", builder["mul"](env.get("_777"), env.get("_765"), {"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_mul_sin_449"}));
  env.set("_780", builder["mul"](env.get("_778"), env.get("_779"), {"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_mul_sign_450"}));
  env.set("_781", builder.reshape(env.get("_780"), [1,sequence_length,32,64]));
  env.set("_786", builder["add"](env.get("_785"), env.get("_781"), {"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_457"}));
  env.set("_787", builder.reshape(env.get("_786"), [1,sequence_length,2048]));
  env.set("_788", builder.reshape(env.get("_787"), [1,sequence_length,32,64]));
  env.set("_789", builder["transpose"](env.get("_788"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/query/transpose_460","permutation":[0,2,1,3]}));
  env.set("Inserted_156", builder.cast(env.get("_598"), "uint8"));
  env.set("_719", builder["where"](env.get("Inserted_156"), env.get("_599"), env.get("_597"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/scatter/where_387"}));
  env.set("_720", builder["add"](env.get("_601"), env.get("_719"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/right_constant/add_389"}));
  env.set("_721", builder.concat([env.get("_603"), env.get("_720")], 2, {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_390"}));
  env.set("_722", builder.reshape(env.get("_721"), [1,sequence_length,4,3]));
  env.set("Inserted_187", builder.cast(env.get("_722"), "int64"));
  env.set("Inserted_189", builder["max"](env.get("Inserted_187"), env.get("Inserted_188"), {"label":"Inserted_Max_432"}));
  env.set("Inserted_191", builder["min"](env.get("Inserted_189"), env.get("Inserted_190"), {"label":"Inserted_Min_433"}));
  env.set("Inserted_180", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_423","maxValue":2047,"minValue":-2048}));
  env.set("_752", builder["gather"](env.get("_641"), env.get("Inserted_180"), {"axis":0,"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_gather_cos_422"}));
  env.set("_753", builder.reshape(env.get("_752"), [1,sequence_length,1,1,32]));
  env.set("_739", builder.dequantizeLinear(env.get("_736"), env.get("_737"), env.get("_738"), {"axis":2,"blockSize":32,"label":"/model/layers.1/attn/k_proj/MatMul_Q4_dequantizeLinear_411"}));
  env.set("_740", builder.reshape(env.get("_739"), [256,2048]));
  env.set("_741", builder["transpose"](env.get("_740"), {"label":"/model/layers.1/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_413","permutation":[1,0]}));
  env.set("_742", builder["matmul"](env.get("_716"), env.get("_741"), {"label":"/model/layers.1/attn/k_proj/MatMul_Q4_matmul_414"}));
  env.set("_743", builder.reshape(env.get("_742"), [1,sequence_length,4,64]));
  env.set("_744", builder.reshape(env.get("_743"), [1,sequence_length,4,2,32]));
  env.set("_754", builder["mul"](env.get("_744"), env.get("_753"), {"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_mul_cos_425"}));
  env.set("_755", builder.reshape(env.get("_754"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_744"), 2, {"axis":3,"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_split_partial_input0_417"});
    env.set("_745", tmp[0]);
    env.set("_746", tmp[1]);
  }
  env.set("_747", builder.concat([env.get("_746"), env.get("_745")], 3, {"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_concat_partial_input0_418"}));
  env.set("Inserted_171", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_409","maxValue":2047,"minValue":-2048}));
  env.set("_734", builder["gather"](env.get("_621"), env.get("Inserted_171"), {"axis":0,"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_gather_sin_408"}));
  env.set("_735", builder.reshape(env.get("_734"), [1,sequence_length,1,1,32]));
  env.set("_748", builder["mul"](env.get("_747"), env.get("_735"), {"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_mul_sin_419"}));
  env.set("_750", builder["mul"](env.get("_748"), env.get("_749"), {"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_mul_sign_420"}));
  env.set("_751", builder.reshape(env.get("_750"), [1,sequence_length,4,64]));
  env.set("_756", builder["add"](env.get("_755"), env.get("_751"), {"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_427"}));
  env.set("_757", builder.reshape(env.get("_756"), [1,sequence_length,256]));
  env.set("_758", builder.reshape(env.get("_757"), [1,sequence_length,4,64]));
  env.set("present_1_key_2", builder["scatterND"](env.get("past_key_values_1_key_759"), env.get("Inserted_191"), env.get("_758"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/present_key/ScatterND_430"}));
  env.set("_760", builder.reshape(env.get("present_1_key_2"), [1,4,1,past_sequence_length,64]));
  env.set("_761", builder.expand(env.get("_760"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/true_present_key/expand_435"}));
  env.set("_762", builder.reshape(env.get("_761"), [1,32,past_sequence_length,64]));
  env.set("_763", builder["transpose"](env.get("_762"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/present_key/transpose_437","permutation":[0,1,3,2]}));
  env.set("_790", builder["matmul"](env.get("_789"), env.get("_763"), {"label":"/model/layers.1/attn/GroupQueryAttention_/Attention/qkv/matmul_1_461"}));
  env.set("_791", builder["mul"](env.get("_790"), env.get("_681"), {"label":"/model/layers.1/attn/GroupQueryAttention_/Attention/qkv/div_462"}));
  env.set("_730", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_402"}));
  env.set("_731", builder.cumulativeSum(env.get("_730"), 3, {"exclusive":true,"label":"/model/layers.1/attn/GroupQueryAttention_range_of_mask_shape_403"}));
  env.set("_727", builder["add"](env.get("_610"), env.get("_719"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/attn_mask/add_399"}));
  env.set("_728", builder.expand(env.get("_727"), [past_sequence_length,sequence_length], {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/expand_neq_right_400"}));
  env.set("_729", builder["transpose"](env.get("_728"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/neq_right/transpose_401","permutation":[1,0]}));
  env.set("Inserted_169", builder["lesser"](env.get("_731"), env.get("_729"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_404"}));
  env.set("_732", builder.cast(env.get("Inserted_169"), "uint8"));
  env.set("Inserted_170", builder.cast(env.get("_732"), "uint8"));
  env.set("_733", builder["where"](env.get("Inserted_170"), env.get("_618"), env.get("_619"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/attn_mask/where_406"}));
  env.set("_792", builder["add"](env.get("_791"), env.get("_733"), {"label":"/model/layers.1/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_463"}));
  env.set("_793", builder["softmax"](env.get("_792"), 3));
  env.set("Inserted_158", builder.cast(env.get("_722"), "int64"));
  env.set("Inserted_160", builder["max"](env.get("Inserted_158"), env.get("Inserted_159"), {"label":"Inserted_Max_394"}));
  env.set("Inserted_162", builder["min"](env.get("Inserted_160"), env.get("Inserted_161"), {"label":"Inserted_Min_395"}));
  env.set("_552", builder.dequantizeLinear(env.get("_549"), env.get("_550"), env.get("_551"), {"axis":2,"blockSize":32,"label":"/model/layers.1/attn/v_proj/MatMul_Q4_dequantizeLinear_252"}));
  env.set("_553", builder.reshape(env.get("_552"), [256,2048]));
  env.set("_554", builder["transpose"](env.get("_553"), {"label":"/model/layers.1/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_254","permutation":[1,0]}));
  env.set("_717", builder["matmul"](env.get("_716"), env.get("_554"), {"label":"/model/layers.1/attn/v_proj/MatMul_Q4_matmul_385"}));
  env.set("_718", builder.reshape(env.get("_717"), [1,sequence_length,4,64]));
  env.set("present_1_value_3", builder["scatterND"](env.get("past_key_values_1_value_723"), env.get("Inserted_162"), env.get("_718"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/present_value/ScatterND_392"}));
  env.set("_724", builder.reshape(env.get("present_1_value_3"), [1,4,1,past_sequence_length,64]));
  env.set("_725", builder.expand(env.get("_724"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/true_present_value/expand_397"}));
  env.set("_726", builder.reshape(env.get("_725"), [1,32,past_sequence_length,64]));
  env.set("_794", builder["matmul"](env.get("_793"), env.get("_726"), {"label":"/model/layers.1/attn/GroupQueryAttention_/Attention/qkv/matmul_2_465"}));
  env.set("_795", builder["transpose"](env.get("_794"), {"label":"/model/layers.1/attn/GroupQueryAttention_/Attention/qkv/transpose_466","permutation":[0,2,1,3]}));
  env.set("_796", builder.reshape(env.get("_795"), [1,sequence_length,2048]));
  env.set("_546", builder.dequantizeLinear(env.get("_543"), env.get("_544"), env.get("_545"), {"axis":2,"blockSize":32,"label":"/model/layers.1/attn/o_proj/MatMul_Q4_dequantizeLinear_249"}));
  env.set("_547", builder.reshape(env.get("_546"), [2048,2048]));
  env.set("_548", builder["transpose"](env.get("_547"), {"label":"/model/layers.1/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_251","permutation":[1,0]}));
  env.set("_797", builder["matmul"](env.get("_796"), env.get("_548"), {"label":"/model/layers.1/attn/o_proj/MatMul_Q4_matmul_468"}));
  env.set("_798", builder["add"](env.get("_709"), env.get("_797"), {"label":"/model/layers.1/post_attention_layernorm/SkipLayerNorm_add_skip_469"}));
  env.set("_799", builder["pow"](env.get("_798"), env.get("_582"), {"label":"/model/layers.1/post_attention_layernorm/SkipLayerNorm_pow_470"}));
  env.set("_800", builder["reduceMean"](env.get("_799"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.1/post_attention_layernorm/SkipLayerNorm_reduceMean_471"}));
  env.set("_801", builder["add"](env.get("_800"), env.get("_585"), {"label":"/model/layers.1/post_attention_layernorm/SkipLayerNorm_add_472"}));
  env.set("_802", builder["sqrt"](env.get("_801"), {"label":"/model/layers.1/post_attention_layernorm/SkipLayerNorm_sqrt_473"}));
  env.set("_803", builder["div"](env.get("_798"), env.get("_802"), {"label":"/model/layers.1/post_attention_layernorm/SkipLayerNorm_div_474"}));
  env.set("_805", builder["mul"](env.get("_804"), env.get("_803"), {"label":"/model/layers.1/post_attention_layernorm/SkipLayerNorm_mul_475"}));
  env.set("_813", builder["matmul"](env.get("_805"), env.get("_812"), {"label":"/model/layers.1/mlp/gate_proj/MatMul_Q4_matmul_480"}));
  env.set("_814", builder["sigmoid"](env.get("_813"), {"label":"/model/layers.1/mlp/act_fn/Sigmoid_481"}));
  env.set("_815", builder["mul"](env.get("_813"), env.get("_814"), {"label":"/model/layers.1/mlp/act_fn/Mul_482"}));
  env.set("_540", builder.dequantizeLinear(env.get("_537"), env.get("_538"), env.get("_539"), {"axis":2,"blockSize":32,"label":"/model/layers.1/mlp/up_proj/MatMul_Q4_dequantizeLinear_246"}));
  env.set("_541", builder.reshape(env.get("_540"), [5632,2048]));
  env.set("_542", builder["transpose"](env.get("_541"), {"label":"/model/layers.1/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_248","permutation":[1,0]}));
  env.set("_806", builder["matmul"](env.get("_805"), env.get("_542"), {"label":"/model/layers.1/mlp/up_proj/MatMul_Q4_matmul_476"}));
  env.set("_816", builder["mul"](env.get("_815"), env.get("_806"), {"label":"/model/layers.1/mlp/Mul_483"}));
  env.set("_534", builder.dequantizeLinear(env.get("_531"), env.get("_532"), env.get("_533"), {"axis":2,"blockSize":32,"label":"/model/layers.1/mlp/down_proj/MatMul_Q4_dequantizeLinear_243"}));
  env.set("_535", builder.reshape(env.get("_534"), [2048,5632]));
  env.set("_536", builder["transpose"](env.get("_535"), {"label":"/model/layers.1/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_245","permutation":[1,0]}));
  env.set("_817", builder["matmul"](env.get("_816"), env.get("_536"), {"label":"/model/layers.1/mlp/down_proj/MatMul_Q4_matmul_484"}));
  env.set("_818", builder["add"](env.get("_798"), env.get("_817"), {"label":"/model/layers.2/input_layernorm/SkipLayerNorm_add_skip_485"}));
  env.set("_819", builder["pow"](env.get("_818"), env.get("_582"), {"label":"/model/layers.2/input_layernorm/SkipLayerNorm_pow_486"}));
  env.set("_820", builder["reduceMean"](env.get("_819"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.2/input_layernorm/SkipLayerNorm_reduceMean_487"}));
  env.set("_821", builder["add"](env.get("_820"), env.get("_585"), {"label":"/model/layers.2/input_layernorm/SkipLayerNorm_add_488"}));
  env.set("_822", builder["sqrt"](env.get("_821"), {"label":"/model/layers.2/input_layernorm/SkipLayerNorm_sqrt_489"}));
  env.set("_823", builder["div"](env.get("_818"), env.get("_822"), {"label":"/model/layers.2/input_layernorm/SkipLayerNorm_div_490"}));
  env.set("_825", builder["mul"](env.get("_824"), env.get("_823"), {"label":"/model/layers.2/input_layernorm/SkipLayerNorm_mul_491"}));
  env.set("_881", builder["matmul"](env.get("_825"), env.get("_880"), {"label":"/model/layers.2/attn/q_proj/MatMul_Q4_matmul_551"}));
  env.set("_882", builder.reshape(env.get("_881"), [1,sequence_length,32,64]));
  env.set("_883", builder.reshape(env.get("_882"), [1,sequence_length,32,2,32]));
  env.set("_893", builder["mul"](env.get("_883"), env.get("_892"), {"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_mul_cos_562"}));
  env.set("_894", builder.reshape(env.get("_893"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_883"), 2, {"axis":3,"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_split_partial_input0_554"});
    env.set("_884", tmp[0]);
    env.set("_885", tmp[1]);
  }
  env.set("_886", builder.concat([env.get("_885"), env.get("_884")], 3, {"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_concat_partial_input0_555"}));
  env.set("Inserted_255", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_546","maxValue":2047,"minValue":-2048}));
  env.set("_873", builder["gather"](env.get("_621"), env.get("Inserted_255"), {"axis":0,"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_gather_sin_545"}));
  env.set("_874", builder.reshape(env.get("_873"), [1,sequence_length,1,1,32]));
  env.set("_887", builder["mul"](env.get("_886"), env.get("_874"), {"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_mul_sin_556"}));
  env.set("_889", builder["mul"](env.get("_887"), env.get("_888"), {"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_mul_sign_557"}));
  env.set("_890", builder.reshape(env.get("_889"), [1,sequence_length,32,64]));
  env.set("_895", builder["add"](env.get("_894"), env.get("_890"), {"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_564"}));
  env.set("_896", builder.reshape(env.get("_895"), [1,sequence_length,2048]));
  env.set("_897", builder.reshape(env.get("_896"), [1,sequence_length,32,64]));
  env.set("_898", builder["transpose"](env.get("_897"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/query/transpose_567","permutation":[0,2,1,3]}));
  env.set("Inserted_216", builder.cast(env.get("_598"), "uint8"));
  env.set("_828", builder["where"](env.get("Inserted_216"), env.get("_599"), env.get("_597"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/scatter/where_494"}));
  env.set("_829", builder["add"](env.get("_601"), env.get("_828"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/right_constant/add_496"}));
  env.set("_830", builder.concat([env.get("_603"), env.get("_829")], 2, {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_497"}));
  env.set("_831", builder.reshape(env.get("_830"), [1,sequence_length,4,3]));
  env.set("Inserted_247", builder.cast(env.get("_831"), "int64"));
  env.set("Inserted_249", builder["max"](env.get("Inserted_247"), env.get("Inserted_248"), {"label":"Inserted_Max_539"}));
  env.set("Inserted_251", builder["min"](env.get("Inserted_249"), env.get("Inserted_250"), {"label":"Inserted_Min_540"}));
  env.set("Inserted_240", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_530","maxValue":2047,"minValue":-2048}));
  env.set("_861", builder["gather"](env.get("_641"), env.get("Inserted_240"), {"axis":0,"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_gather_cos_529"}));
  env.set("_862", builder.reshape(env.get("_861"), [1,sequence_length,1,1,32]));
  env.set("_848", builder.dequantizeLinear(env.get("_845"), env.get("_846"), env.get("_847"), {"axis":2,"blockSize":32,"label":"/model/layers.2/attn/k_proj/MatMul_Q4_dequantizeLinear_518"}));
  env.set("_849", builder.reshape(env.get("_848"), [256,2048]));
  env.set("_850", builder["transpose"](env.get("_849"), {"label":"/model/layers.2/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_520","permutation":[1,0]}));
  env.set("_851", builder["matmul"](env.get("_825"), env.get("_850"), {"label":"/model/layers.2/attn/k_proj/MatMul_Q4_matmul_521"}));
  env.set("_852", builder.reshape(env.get("_851"), [1,sequence_length,4,64]));
  env.set("_853", builder.reshape(env.get("_852"), [1,sequence_length,4,2,32]));
  env.set("_863", builder["mul"](env.get("_853"), env.get("_862"), {"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_mul_cos_532"}));
  env.set("_864", builder.reshape(env.get("_863"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_853"), 2, {"axis":3,"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_split_partial_input0_524"});
    env.set("_854", tmp[0]);
    env.set("_855", tmp[1]);
  }
  env.set("_856", builder.concat([env.get("_855"), env.get("_854")], 3, {"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_concat_partial_input0_525"}));
  env.set("Inserted_231", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_516","maxValue":2047,"minValue":-2048}));
  env.set("_843", builder["gather"](env.get("_621"), env.get("Inserted_231"), {"axis":0,"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_gather_sin_515"}));
  env.set("_844", builder.reshape(env.get("_843"), [1,sequence_length,1,1,32]));
  env.set("_857", builder["mul"](env.get("_856"), env.get("_844"), {"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_mul_sin_526"}));
  env.set("_859", builder["mul"](env.get("_857"), env.get("_858"), {"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_mul_sign_527"}));
  env.set("_860", builder.reshape(env.get("_859"), [1,sequence_length,4,64]));
  env.set("_865", builder["add"](env.get("_864"), env.get("_860"), {"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_534"}));
  env.set("_866", builder.reshape(env.get("_865"), [1,sequence_length,256]));
  env.set("_867", builder.reshape(env.get("_866"), [1,sequence_length,4,64]));
  env.set("present_2_key_4", builder["scatterND"](env.get("past_key_values_2_key_868"), env.get("Inserted_251"), env.get("_867"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/present_key/ScatterND_537"}));
  env.set("_869", builder.reshape(env.get("present_2_key_4"), [1,4,1,past_sequence_length,64]));
  env.set("_870", builder.expand(env.get("_869"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/true_present_key/expand_542"}));
  env.set("_871", builder.reshape(env.get("_870"), [1,32,past_sequence_length,64]));
  env.set("_872", builder["transpose"](env.get("_871"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/present_key/transpose_544","permutation":[0,1,3,2]}));
  env.set("_899", builder["matmul"](env.get("_898"), env.get("_872"), {"label":"/model/layers.2/attn/GroupQueryAttention_/Attention/qkv/matmul_1_568"}));
  env.set("_900", builder["mul"](env.get("_899"), env.get("_681"), {"label":"/model/layers.2/attn/GroupQueryAttention_/Attention/qkv/div_569"}));
  env.set("_839", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_509"}));
  env.set("_840", builder.cumulativeSum(env.get("_839"), 3, {"exclusive":true,"label":"/model/layers.2/attn/GroupQueryAttention_range_of_mask_shape_510"}));
  env.set("_836", builder["add"](env.get("_610"), env.get("_828"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/attn_mask/add_506"}));
  env.set("_837", builder.expand(env.get("_836"), [past_sequence_length,sequence_length], {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/expand_neq_right_507"}));
  env.set("_838", builder["transpose"](env.get("_837"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/neq_right/transpose_508","permutation":[1,0]}));
  env.set("Inserted_229", builder["lesser"](env.get("_840"), env.get("_838"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_511"}));
  env.set("_841", builder.cast(env.get("Inserted_229"), "uint8"));
  env.set("Inserted_230", builder.cast(env.get("_841"), "uint8"));
  env.set("_842", builder["where"](env.get("Inserted_230"), env.get("_618"), env.get("_619"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/attn_mask/where_513"}));
  env.set("_901", builder["add"](env.get("_900"), env.get("_842"), {"label":"/model/layers.2/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_570"}));
  env.set("_902", builder["softmax"](env.get("_901"), 3));
  env.set("Inserted_218", builder.cast(env.get("_831"), "int64"));
  env.set("Inserted_220", builder["max"](env.get("Inserted_218"), env.get("Inserted_219"), {"label":"Inserted_Max_501"}));
  env.set("Inserted_222", builder["min"](env.get("Inserted_220"), env.get("Inserted_221"), {"label":"Inserted_Min_502"}));
  env.set("_528", builder.dequantizeLinear(env.get("_525"), env.get("_526"), env.get("_527"), {"axis":2,"blockSize":32,"label":"/model/layers.2/attn/v_proj/MatMul_Q4_dequantizeLinear_240"}));
  env.set("_529", builder.reshape(env.get("_528"), [256,2048]));
  env.set("_530", builder["transpose"](env.get("_529"), {"label":"/model/layers.2/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_242","permutation":[1,0]}));
  env.set("_826", builder["matmul"](env.get("_825"), env.get("_530"), {"label":"/model/layers.2/attn/v_proj/MatMul_Q4_matmul_492"}));
  env.set("_827", builder.reshape(env.get("_826"), [1,sequence_length,4,64]));
  env.set("present_2_value_5", builder["scatterND"](env.get("past_key_values_2_value_832"), env.get("Inserted_222"), env.get("_827"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/present_value/ScatterND_499"}));
  env.set("_833", builder.reshape(env.get("present_2_value_5"), [1,4,1,past_sequence_length,64]));
  env.set("_834", builder.expand(env.get("_833"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/true_present_value/expand_504"}));
  env.set("_835", builder.reshape(env.get("_834"), [1,32,past_sequence_length,64]));
  env.set("_903", builder["matmul"](env.get("_902"), env.get("_835"), {"label":"/model/layers.2/attn/GroupQueryAttention_/Attention/qkv/matmul_2_572"}));
  env.set("_904", builder["transpose"](env.get("_903"), {"label":"/model/layers.2/attn/GroupQueryAttention_/Attention/qkv/transpose_573","permutation":[0,2,1,3]}));
  env.set("_905", builder.reshape(env.get("_904"), [1,sequence_length,2048]));
  env.set("_522", builder.dequantizeLinear(env.get("_519"), env.get("_520"), env.get("_521"), {"axis":2,"blockSize":32,"label":"/model/layers.2/attn/o_proj/MatMul_Q4_dequantizeLinear_237"}));
  env.set("_523", builder.reshape(env.get("_522"), [2048,2048]));
  env.set("_524", builder["transpose"](env.get("_523"), {"label":"/model/layers.2/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_239","permutation":[1,0]}));
  env.set("_906", builder["matmul"](env.get("_905"), env.get("_524"), {"label":"/model/layers.2/attn/o_proj/MatMul_Q4_matmul_575"}));
  env.set("_907", builder["add"](env.get("_818"), env.get("_906"), {"label":"/model/layers.2/post_attention_layernorm/SkipLayerNorm_add_skip_576"}));
  env.set("_908", builder["pow"](env.get("_907"), env.get("_582"), {"label":"/model/layers.2/post_attention_layernorm/SkipLayerNorm_pow_577"}));
  env.set("_909", builder["reduceMean"](env.get("_908"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.2/post_attention_layernorm/SkipLayerNorm_reduceMean_578"}));
  env.set("_910", builder["add"](env.get("_909"), env.get("_585"), {"label":"/model/layers.2/post_attention_layernorm/SkipLayerNorm_add_579"}));
  env.set("_911", builder["sqrt"](env.get("_910"), {"label":"/model/layers.2/post_attention_layernorm/SkipLayerNorm_sqrt_580"}));
  env.set("_912", builder["div"](env.get("_907"), env.get("_911"), {"label":"/model/layers.2/post_attention_layernorm/SkipLayerNorm_div_581"}));
  env.set("_914", builder["mul"](env.get("_913"), env.get("_912"), {"label":"/model/layers.2/post_attention_layernorm/SkipLayerNorm_mul_582"}));
  env.set("_922", builder["matmul"](env.get("_914"), env.get("_921"), {"label":"/model/layers.2/mlp/gate_proj/MatMul_Q4_matmul_587"}));
  env.set("_923", builder["sigmoid"](env.get("_922"), {"label":"/model/layers.2/mlp/act_fn/Sigmoid_588"}));
  env.set("_924", builder["mul"](env.get("_922"), env.get("_923"), {"label":"/model/layers.2/mlp/act_fn/Mul_589"}));
  env.set("_516", builder.dequantizeLinear(env.get("_513"), env.get("_514"), env.get("_515"), {"axis":2,"blockSize":32,"label":"/model/layers.2/mlp/up_proj/MatMul_Q4_dequantizeLinear_234"}));
  env.set("_517", builder.reshape(env.get("_516"), [5632,2048]));
  env.set("_518", builder["transpose"](env.get("_517"), {"label":"/model/layers.2/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_236","permutation":[1,0]}));
  env.set("_915", builder["matmul"](env.get("_914"), env.get("_518"), {"label":"/model/layers.2/mlp/up_proj/MatMul_Q4_matmul_583"}));
  env.set("_925", builder["mul"](env.get("_924"), env.get("_915"), {"label":"/model/layers.2/mlp/Mul_590"}));
  env.set("_510", builder.dequantizeLinear(env.get("_507"), env.get("_508"), env.get("_509"), {"axis":2,"blockSize":32,"label":"/model/layers.2/mlp/down_proj/MatMul_Q4_dequantizeLinear_231"}));
  env.set("_511", builder.reshape(env.get("_510"), [2048,5632]));
  env.set("_512", builder["transpose"](env.get("_511"), {"label":"/model/layers.2/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_233","permutation":[1,0]}));
  env.set("_926", builder["matmul"](env.get("_925"), env.get("_512"), {"label":"/model/layers.2/mlp/down_proj/MatMul_Q4_matmul_591"}));
  env.set("_927", builder["add"](env.get("_907"), env.get("_926"), {"label":"/model/layers.3/input_layernorm/SkipLayerNorm_add_skip_592"}));
  env.set("_928", builder["pow"](env.get("_927"), env.get("_582"), {"label":"/model/layers.3/input_layernorm/SkipLayerNorm_pow_593"}));
  env.set("_929", builder["reduceMean"](env.get("_928"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.3/input_layernorm/SkipLayerNorm_reduceMean_594"}));
  env.set("_930", builder["add"](env.get("_929"), env.get("_585"), {"label":"/model/layers.3/input_layernorm/SkipLayerNorm_add_595"}));
  env.set("_931", builder["sqrt"](env.get("_930"), {"label":"/model/layers.3/input_layernorm/SkipLayerNorm_sqrt_596"}));
  env.set("_932", builder["div"](env.get("_927"), env.get("_931"), {"label":"/model/layers.3/input_layernorm/SkipLayerNorm_div_597"}));
  env.set("_934", builder["mul"](env.get("_933"), env.get("_932"), {"label":"/model/layers.3/input_layernorm/SkipLayerNorm_mul_598"}));
  env.set("_990", builder["matmul"](env.get("_934"), env.get("_989"), {"label":"/model/layers.3/attn/q_proj/MatMul_Q4_matmul_658"}));
  env.set("_991", builder.reshape(env.get("_990"), [1,sequence_length,32,64]));
  env.set("_992", builder.reshape(env.get("_991"), [1,sequence_length,32,2,32]));
  env.set("_1002", builder["mul"](env.get("_992"), env.get("_1001"), {"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_mul_cos_669"}));
  env.set("_1003", builder.reshape(env.get("_1002"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_992"), 2, {"axis":3,"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_split_partial_input0_661"});
    env.set("_993", tmp[0]);
    env.set("_994", tmp[1]);
  }
  env.set("_995", builder.concat([env.get("_994"), env.get("_993")], 3, {"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_concat_partial_input0_662"}));
  env.set("Inserted_315", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_653","maxValue":2047,"minValue":-2048}));
  env.set("_982", builder["gather"](env.get("_621"), env.get("Inserted_315"), {"axis":0,"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_gather_sin_652"}));
  env.set("_983", builder.reshape(env.get("_982"), [1,sequence_length,1,1,32]));
  env.set("_996", builder["mul"](env.get("_995"), env.get("_983"), {"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_mul_sin_663"}));
  env.set("_998", builder["mul"](env.get("_996"), env.get("_997"), {"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_mul_sign_664"}));
  env.set("_999", builder.reshape(env.get("_998"), [1,sequence_length,32,64]));
  env.set("_1004", builder["add"](env.get("_1003"), env.get("_999"), {"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_671"}));
  env.set("_1005", builder.reshape(env.get("_1004"), [1,sequence_length,2048]));
  env.set("_1006", builder.reshape(env.get("_1005"), [1,sequence_length,32,64]));
  env.set("_1007", builder["transpose"](env.get("_1006"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/query/transpose_674","permutation":[0,2,1,3]}));
  env.set("Inserted_276", builder.cast(env.get("_598"), "uint8"));
  env.set("_937", builder["where"](env.get("Inserted_276"), env.get("_599"), env.get("_597"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/scatter/where_601"}));
  env.set("_938", builder["add"](env.get("_601"), env.get("_937"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/right_constant/add_603"}));
  env.set("_939", builder.concat([env.get("_603"), env.get("_938")], 2, {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_604"}));
  env.set("_940", builder.reshape(env.get("_939"), [1,sequence_length,4,3]));
  env.set("Inserted_307", builder.cast(env.get("_940"), "int64"));
  env.set("Inserted_309", builder["max"](env.get("Inserted_307"), env.get("Inserted_308"), {"label":"Inserted_Max_646"}));
  env.set("Inserted_311", builder["min"](env.get("Inserted_309"), env.get("Inserted_310"), {"label":"Inserted_Min_647"}));
  env.set("Inserted_300", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_637","maxValue":2047,"minValue":-2048}));
  env.set("_970", builder["gather"](env.get("_641"), env.get("Inserted_300"), {"axis":0,"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_gather_cos_636"}));
  env.set("_971", builder.reshape(env.get("_970"), [1,sequence_length,1,1,32]));
  env.set("_957", builder.dequantizeLinear(env.get("_954"), env.get("_955"), env.get("_956"), {"axis":2,"blockSize":32,"label":"/model/layers.3/attn/k_proj/MatMul_Q4_dequantizeLinear_625"}));
  env.set("_958", builder.reshape(env.get("_957"), [256,2048]));
  env.set("_959", builder["transpose"](env.get("_958"), {"label":"/model/layers.3/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_627","permutation":[1,0]}));
  env.set("_960", builder["matmul"](env.get("_934"), env.get("_959"), {"label":"/model/layers.3/attn/k_proj/MatMul_Q4_matmul_628"}));
  env.set("_961", builder.reshape(env.get("_960"), [1,sequence_length,4,64]));
  env.set("_962", builder.reshape(env.get("_961"), [1,sequence_length,4,2,32]));
  env.set("_972", builder["mul"](env.get("_962"), env.get("_971"), {"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_mul_cos_639"}));
  env.set("_973", builder.reshape(env.get("_972"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_962"), 2, {"axis":3,"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_split_partial_input0_631"});
    env.set("_963", tmp[0]);
    env.set("_964", tmp[1]);
  }
  env.set("_965", builder.concat([env.get("_964"), env.get("_963")], 3, {"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_concat_partial_input0_632"}));
  env.set("Inserted_291", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_623","maxValue":2047,"minValue":-2048}));
  env.set("_952", builder["gather"](env.get("_621"), env.get("Inserted_291"), {"axis":0,"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_gather_sin_622"}));
  env.set("_953", builder.reshape(env.get("_952"), [1,sequence_length,1,1,32]));
  env.set("_966", builder["mul"](env.get("_965"), env.get("_953"), {"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_mul_sin_633"}));
  env.set("_968", builder["mul"](env.get("_966"), env.get("_967"), {"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_mul_sign_634"}));
  env.set("_969", builder.reshape(env.get("_968"), [1,sequence_length,4,64]));
  env.set("_974", builder["add"](env.get("_973"), env.get("_969"), {"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_641"}));
  env.set("_975", builder.reshape(env.get("_974"), [1,sequence_length,256]));
  env.set("_976", builder.reshape(env.get("_975"), [1,sequence_length,4,64]));
  env.set("present_3_key_6", builder["scatterND"](env.get("past_key_values_3_key_977"), env.get("Inserted_311"), env.get("_976"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/present_key/ScatterND_644"}));
  env.set("_978", builder.reshape(env.get("present_3_key_6"), [1,4,1,past_sequence_length,64]));
  env.set("_979", builder.expand(env.get("_978"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/true_present_key/expand_649"}));
  env.set("_980", builder.reshape(env.get("_979"), [1,32,past_sequence_length,64]));
  env.set("_981", builder["transpose"](env.get("_980"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/present_key/transpose_651","permutation":[0,1,3,2]}));
  env.set("_1008", builder["matmul"](env.get("_1007"), env.get("_981"), {"label":"/model/layers.3/attn/GroupQueryAttention_/Attention/qkv/matmul_1_675"}));
  env.set("_1009", builder["mul"](env.get("_1008"), env.get("_681"), {"label":"/model/layers.3/attn/GroupQueryAttention_/Attention/qkv/div_676"}));
  env.set("_948", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_616"}));
  env.set("_949", builder.cumulativeSum(env.get("_948"), 3, {"exclusive":true,"label":"/model/layers.3/attn/GroupQueryAttention_range_of_mask_shape_617"}));
  env.set("_945", builder["add"](env.get("_610"), env.get("_937"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/attn_mask/add_613"}));
  env.set("_946", builder.expand(env.get("_945"), [past_sequence_length,sequence_length], {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/expand_neq_right_614"}));
  env.set("_947", builder["transpose"](env.get("_946"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/neq_right/transpose_615","permutation":[1,0]}));
  env.set("Inserted_289", builder["lesser"](env.get("_949"), env.get("_947"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_618"}));
  env.set("_950", builder.cast(env.get("Inserted_289"), "uint8"));
  env.set("Inserted_290", builder.cast(env.get("_950"), "uint8"));
  env.set("_951", builder["where"](env.get("Inserted_290"), env.get("_618"), env.get("_619"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/attn_mask/where_620"}));
  env.set("_1010", builder["add"](env.get("_1009"), env.get("_951"), {"label":"/model/layers.3/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_677"}));
  env.set("_1011", builder["softmax"](env.get("_1010"), 3));
  env.set("Inserted_278", builder.cast(env.get("_940"), "int64"));
  env.set("Inserted_280", builder["max"](env.get("Inserted_278"), env.get("Inserted_279"), {"label":"Inserted_Max_608"}));
  env.set("Inserted_282", builder["min"](env.get("Inserted_280"), env.get("Inserted_281"), {"label":"Inserted_Min_609"}));
  env.set("_504", builder.dequantizeLinear(env.get("_501"), env.get("_502"), env.get("_503"), {"axis":2,"blockSize":32,"label":"/model/layers.3/attn/v_proj/MatMul_Q4_dequantizeLinear_228"}));
  env.set("_505", builder.reshape(env.get("_504"), [256,2048]));
  env.set("_506", builder["transpose"](env.get("_505"), {"label":"/model/layers.3/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_230","permutation":[1,0]}));
  env.set("_935", builder["matmul"](env.get("_934"), env.get("_506"), {"label":"/model/layers.3/attn/v_proj/MatMul_Q4_matmul_599"}));
  env.set("_936", builder.reshape(env.get("_935"), [1,sequence_length,4,64]));
  env.set("present_3_value_7", builder["scatterND"](env.get("past_key_values_3_value_941"), env.get("Inserted_282"), env.get("_936"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/present_value/ScatterND_606"}));
  env.set("_942", builder.reshape(env.get("present_3_value_7"), [1,4,1,past_sequence_length,64]));
  env.set("_943", builder.expand(env.get("_942"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/true_present_value/expand_611"}));
  env.set("_944", builder.reshape(env.get("_943"), [1,32,past_sequence_length,64]));
  env.set("_1012", builder["matmul"](env.get("_1011"), env.get("_944"), {"label":"/model/layers.3/attn/GroupQueryAttention_/Attention/qkv/matmul_2_679"}));
  env.set("_1013", builder["transpose"](env.get("_1012"), {"label":"/model/layers.3/attn/GroupQueryAttention_/Attention/qkv/transpose_680","permutation":[0,2,1,3]}));
  env.set("_1014", builder.reshape(env.get("_1013"), [1,sequence_length,2048]));
  env.set("_498", builder.dequantizeLinear(env.get("_495"), env.get("_496"), env.get("_497"), {"axis":2,"blockSize":32,"label":"/model/layers.3/attn/o_proj/MatMul_Q4_dequantizeLinear_225"}));
  env.set("_499", builder.reshape(env.get("_498"), [2048,2048]));
  env.set("_500", builder["transpose"](env.get("_499"), {"label":"/model/layers.3/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_227","permutation":[1,0]}));
  env.set("_1015", builder["matmul"](env.get("_1014"), env.get("_500"), {"label":"/model/layers.3/attn/o_proj/MatMul_Q4_matmul_682"}));
  env.set("_1016", builder["add"](env.get("_927"), env.get("_1015"), {"label":"/model/layers.3/post_attention_layernorm/SkipLayerNorm_add_skip_683"}));
  env.set("_1017", builder["pow"](env.get("_1016"), env.get("_582"), {"label":"/model/layers.3/post_attention_layernorm/SkipLayerNorm_pow_684"}));
  env.set("_1018", builder["reduceMean"](env.get("_1017"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.3/post_attention_layernorm/SkipLayerNorm_reduceMean_685"}));
  env.set("_1019", builder["add"](env.get("_1018"), env.get("_585"), {"label":"/model/layers.3/post_attention_layernorm/SkipLayerNorm_add_686"}));
  env.set("_1020", builder["sqrt"](env.get("_1019"), {"label":"/model/layers.3/post_attention_layernorm/SkipLayerNorm_sqrt_687"}));
  env.set("_1021", builder["div"](env.get("_1016"), env.get("_1020"), {"label":"/model/layers.3/post_attention_layernorm/SkipLayerNorm_div_688"}));
  env.set("_1023", builder["mul"](env.get("_1022"), env.get("_1021"), {"label":"/model/layers.3/post_attention_layernorm/SkipLayerNorm_mul_689"}));
  env.set("_1031", builder["matmul"](env.get("_1023"), env.get("_1030"), {"label":"/model/layers.3/mlp/gate_proj/MatMul_Q4_matmul_694"}));
  env.set("_1032", builder["sigmoid"](env.get("_1031"), {"label":"/model/layers.3/mlp/act_fn/Sigmoid_695"}));
  env.set("_1033", builder["mul"](env.get("_1031"), env.get("_1032"), {"label":"/model/layers.3/mlp/act_fn/Mul_696"}));
  env.set("_492", builder.dequantizeLinear(env.get("_489"), env.get("_490"), env.get("_491"), {"axis":2,"blockSize":32,"label":"/model/layers.3/mlp/up_proj/MatMul_Q4_dequantizeLinear_222"}));
  env.set("_493", builder.reshape(env.get("_492"), [5632,2048]));
  env.set("_494", builder["transpose"](env.get("_493"), {"label":"/model/layers.3/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_224","permutation":[1,0]}));
  env.set("_1024", builder["matmul"](env.get("_1023"), env.get("_494"), {"label":"/model/layers.3/mlp/up_proj/MatMul_Q4_matmul_690"}));
  env.set("_1034", builder["mul"](env.get("_1033"), env.get("_1024"), {"label":"/model/layers.3/mlp/Mul_697"}));
  env.set("_486", builder.dequantizeLinear(env.get("_483"), env.get("_484"), env.get("_485"), {"axis":2,"blockSize":32,"label":"/model/layers.3/mlp/down_proj/MatMul_Q4_dequantizeLinear_219"}));
  env.set("_487", builder.reshape(env.get("_486"), [2048,5632]));
  env.set("_488", builder["transpose"](env.get("_487"), {"label":"/model/layers.3/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_221","permutation":[1,0]}));
  env.set("_1035", builder["matmul"](env.get("_1034"), env.get("_488"), {"label":"/model/layers.3/mlp/down_proj/MatMul_Q4_matmul_698"}));
  env.set("_1036", builder["add"](env.get("_1016"), env.get("_1035"), {"label":"/model/layers.4/input_layernorm/SkipLayerNorm_add_skip_699"}));
  env.set("_1037", builder["pow"](env.get("_1036"), env.get("_582"), {"label":"/model/layers.4/input_layernorm/SkipLayerNorm_pow_700"}));
  env.set("_1038", builder["reduceMean"](env.get("_1037"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.4/input_layernorm/SkipLayerNorm_reduceMean_701"}));
  env.set("_1039", builder["add"](env.get("_1038"), env.get("_585"), {"label":"/model/layers.4/input_layernorm/SkipLayerNorm_add_702"}));
  env.set("_1040", builder["sqrt"](env.get("_1039"), {"label":"/model/layers.4/input_layernorm/SkipLayerNorm_sqrt_703"}));
  env.set("_1041", builder["div"](env.get("_1036"), env.get("_1040"), {"label":"/model/layers.4/input_layernorm/SkipLayerNorm_div_704"}));
  env.set("_1043", builder["mul"](env.get("_1042"), env.get("_1041"), {"label":"/model/layers.4/input_layernorm/SkipLayerNorm_mul_705"}));
  env.set("_1099", builder["matmul"](env.get("_1043"), env.get("_1098"), {"label":"/model/layers.4/attn/q_proj/MatMul_Q4_matmul_765"}));
  env.set("_1100", builder.reshape(env.get("_1099"), [1,sequence_length,32,64]));
  env.set("_1101", builder.reshape(env.get("_1100"), [1,sequence_length,32,2,32]));
  env.set("_1111", builder["mul"](env.get("_1101"), env.get("_1110"), {"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_mul_cos_776"}));
  env.set("_1112", builder.reshape(env.get("_1111"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_1101"), 2, {"axis":3,"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_split_partial_input0_768"});
    env.set("_1102", tmp[0]);
    env.set("_1103", tmp[1]);
  }
  env.set("_1104", builder.concat([env.get("_1103"), env.get("_1102")], 3, {"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_concat_partial_input0_769"}));
  env.set("Inserted_375", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_760","maxValue":2047,"minValue":-2048}));
  env.set("_1091", builder["gather"](env.get("_621"), env.get("Inserted_375"), {"axis":0,"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_gather_sin_759"}));
  env.set("_1092", builder.reshape(env.get("_1091"), [1,sequence_length,1,1,32]));
  env.set("_1105", builder["mul"](env.get("_1104"), env.get("_1092"), {"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_mul_sin_770"}));
  env.set("_1107", builder["mul"](env.get("_1105"), env.get("_1106"), {"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_mul_sign_771"}));
  env.set("_1108", builder.reshape(env.get("_1107"), [1,sequence_length,32,64]));
  env.set("_1113", builder["add"](env.get("_1112"), env.get("_1108"), {"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_778"}));
  env.set("_1114", builder.reshape(env.get("_1113"), [1,sequence_length,2048]));
  env.set("_1115", builder.reshape(env.get("_1114"), [1,sequence_length,32,64]));
  env.set("_1116", builder["transpose"](env.get("_1115"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/query/transpose_781","permutation":[0,2,1,3]}));
  env.set("Inserted_336", builder.cast(env.get("_598"), "uint8"));
  env.set("_1046", builder["where"](env.get("Inserted_336"), env.get("_599"), env.get("_597"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/scatter/where_708"}));
  env.set("_1047", builder["add"](env.get("_601"), env.get("_1046"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/right_constant/add_710"}));
  env.set("_1048", builder.concat([env.get("_603"), env.get("_1047")], 2, {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_711"}));
  env.set("_1049", builder.reshape(env.get("_1048"), [1,sequence_length,4,3]));
  env.set("Inserted_367", builder.cast(env.get("_1049"), "int64"));
  env.set("Inserted_369", builder["max"](env.get("Inserted_367"), env.get("Inserted_368"), {"label":"Inserted_Max_753"}));
  env.set("Inserted_371", builder["min"](env.get("Inserted_369"), env.get("Inserted_370"), {"label":"Inserted_Min_754"}));
  env.set("Inserted_360", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_744","maxValue":2047,"minValue":-2048}));
  env.set("_1079", builder["gather"](env.get("_641"), env.get("Inserted_360"), {"axis":0,"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_gather_cos_743"}));
  env.set("_1080", builder.reshape(env.get("_1079"), [1,sequence_length,1,1,32]));
  env.set("_1066", builder.dequantizeLinear(env.get("_1063"), env.get("_1064"), env.get("_1065"), {"axis":2,"blockSize":32,"label":"/model/layers.4/attn/k_proj/MatMul_Q4_dequantizeLinear_732"}));
  env.set("_1067", builder.reshape(env.get("_1066"), [256,2048]));
  env.set("_1068", builder["transpose"](env.get("_1067"), {"label":"/model/layers.4/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_734","permutation":[1,0]}));
  env.set("_1069", builder["matmul"](env.get("_1043"), env.get("_1068"), {"label":"/model/layers.4/attn/k_proj/MatMul_Q4_matmul_735"}));
  env.set("_1070", builder.reshape(env.get("_1069"), [1,sequence_length,4,64]));
  env.set("_1071", builder.reshape(env.get("_1070"), [1,sequence_length,4,2,32]));
  env.set("_1081", builder["mul"](env.get("_1071"), env.get("_1080"), {"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_mul_cos_746"}));
  env.set("_1082", builder.reshape(env.get("_1081"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_1071"), 2, {"axis":3,"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_split_partial_input0_738"});
    env.set("_1072", tmp[0]);
    env.set("_1073", tmp[1]);
  }
  env.set("_1074", builder.concat([env.get("_1073"), env.get("_1072")], 3, {"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_concat_partial_input0_739"}));
  env.set("Inserted_351", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_730","maxValue":2047,"minValue":-2048}));
  env.set("_1061", builder["gather"](env.get("_621"), env.get("Inserted_351"), {"axis":0,"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_gather_sin_729"}));
  env.set("_1062", builder.reshape(env.get("_1061"), [1,sequence_length,1,1,32]));
  env.set("_1075", builder["mul"](env.get("_1074"), env.get("_1062"), {"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_mul_sin_740"}));
  env.set("_1077", builder["mul"](env.get("_1075"), env.get("_1076"), {"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_mul_sign_741"}));
  env.set("_1078", builder.reshape(env.get("_1077"), [1,sequence_length,4,64]));
  env.set("_1083", builder["add"](env.get("_1082"), env.get("_1078"), {"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_748"}));
  env.set("_1084", builder.reshape(env.get("_1083"), [1,sequence_length,256]));
  env.set("_1085", builder.reshape(env.get("_1084"), [1,sequence_length,4,64]));
  env.set("present_4_key_8", builder["scatterND"](env.get("past_key_values_4_key_1086"), env.get("Inserted_371"), env.get("_1085"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/present_key/ScatterND_751"}));
  env.set("_1087", builder.reshape(env.get("present_4_key_8"), [1,4,1,past_sequence_length,64]));
  env.set("_1088", builder.expand(env.get("_1087"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/true_present_key/expand_756"}));
  env.set("_1089", builder.reshape(env.get("_1088"), [1,32,past_sequence_length,64]));
  env.set("_1090", builder["transpose"](env.get("_1089"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/present_key/transpose_758","permutation":[0,1,3,2]}));
  env.set("_1117", builder["matmul"](env.get("_1116"), env.get("_1090"), {"label":"/model/layers.4/attn/GroupQueryAttention_/Attention/qkv/matmul_1_782"}));
  env.set("_1118", builder["mul"](env.get("_1117"), env.get("_681"), {"label":"/model/layers.4/attn/GroupQueryAttention_/Attention/qkv/div_783"}));
  env.set("_1057", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_723"}));
  env.set("_1058", builder.cumulativeSum(env.get("_1057"), 3, {"exclusive":true,"label":"/model/layers.4/attn/GroupQueryAttention_range_of_mask_shape_724"}));
  env.set("_1054", builder["add"](env.get("_610"), env.get("_1046"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/attn_mask/add_720"}));
  env.set("_1055", builder.expand(env.get("_1054"), [past_sequence_length,sequence_length], {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/expand_neq_right_721"}));
  env.set("_1056", builder["transpose"](env.get("_1055"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/neq_right/transpose_722","permutation":[1,0]}));
  env.set("Inserted_349", builder["lesser"](env.get("_1058"), env.get("_1056"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_725"}));
  env.set("_1059", builder.cast(env.get("Inserted_349"), "uint8"));
  env.set("Inserted_350", builder.cast(env.get("_1059"), "uint8"));
  env.set("_1060", builder["where"](env.get("Inserted_350"), env.get("_618"), env.get("_619"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/attn_mask/where_727"}));
  env.set("_1119", builder["add"](env.get("_1118"), env.get("_1060"), {"label":"/model/layers.4/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_784"}));
  env.set("_1120", builder["softmax"](env.get("_1119"), 3));
  env.set("Inserted_338", builder.cast(env.get("_1049"), "int64"));
  env.set("Inserted_340", builder["max"](env.get("Inserted_338"), env.get("Inserted_339"), {"label":"Inserted_Max_715"}));
  env.set("Inserted_342", builder["min"](env.get("Inserted_340"), env.get("Inserted_341"), {"label":"Inserted_Min_716"}));
  env.set("_480", builder.dequantizeLinear(env.get("_477"), env.get("_478"), env.get("_479"), {"axis":2,"blockSize":32,"label":"/model/layers.4/attn/v_proj/MatMul_Q4_dequantizeLinear_216"}));
  env.set("_481", builder.reshape(env.get("_480"), [256,2048]));
  env.set("_482", builder["transpose"](env.get("_481"), {"label":"/model/layers.4/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_218","permutation":[1,0]}));
  env.set("_1044", builder["matmul"](env.get("_1043"), env.get("_482"), {"label":"/model/layers.4/attn/v_proj/MatMul_Q4_matmul_706"}));
  env.set("_1045", builder.reshape(env.get("_1044"), [1,sequence_length,4,64]));
  env.set("present_4_value_9", builder["scatterND"](env.get("past_key_values_4_value_1050"), env.get("Inserted_342"), env.get("_1045"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/present_value/ScatterND_713"}));
  env.set("_1051", builder.reshape(env.get("present_4_value_9"), [1,4,1,past_sequence_length,64]));
  env.set("_1052", builder.expand(env.get("_1051"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/true_present_value/expand_718"}));
  env.set("_1053", builder.reshape(env.get("_1052"), [1,32,past_sequence_length,64]));
  env.set("_1121", builder["matmul"](env.get("_1120"), env.get("_1053"), {"label":"/model/layers.4/attn/GroupQueryAttention_/Attention/qkv/matmul_2_786"}));
  env.set("_1122", builder["transpose"](env.get("_1121"), {"label":"/model/layers.4/attn/GroupQueryAttention_/Attention/qkv/transpose_787","permutation":[0,2,1,3]}));
  env.set("_1123", builder.reshape(env.get("_1122"), [1,sequence_length,2048]));
  env.set("_474", builder.dequantizeLinear(env.get("_471"), env.get("_472"), env.get("_473"), {"axis":2,"blockSize":32,"label":"/model/layers.4/attn/o_proj/MatMul_Q4_dequantizeLinear_213"}));
  env.set("_475", builder.reshape(env.get("_474"), [2048,2048]));
  env.set("_476", builder["transpose"](env.get("_475"), {"label":"/model/layers.4/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_215","permutation":[1,0]}));
  env.set("_1124", builder["matmul"](env.get("_1123"), env.get("_476"), {"label":"/model/layers.4/attn/o_proj/MatMul_Q4_matmul_789"}));
  env.set("_1125", builder["add"](env.get("_1036"), env.get("_1124"), {"label":"/model/layers.4/post_attention_layernorm/SkipLayerNorm_add_skip_790"}));
  env.set("_1126", builder["pow"](env.get("_1125"), env.get("_582"), {"label":"/model/layers.4/post_attention_layernorm/SkipLayerNorm_pow_791"}));
  env.set("_1127", builder["reduceMean"](env.get("_1126"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.4/post_attention_layernorm/SkipLayerNorm_reduceMean_792"}));
  env.set("_1128", builder["add"](env.get("_1127"), env.get("_585"), {"label":"/model/layers.4/post_attention_layernorm/SkipLayerNorm_add_793"}));
  env.set("_1129", builder["sqrt"](env.get("_1128"), {"label":"/model/layers.4/post_attention_layernorm/SkipLayerNorm_sqrt_794"}));
  env.set("_1130", builder["div"](env.get("_1125"), env.get("_1129"), {"label":"/model/layers.4/post_attention_layernorm/SkipLayerNorm_div_795"}));
  env.set("_1132", builder["mul"](env.get("_1131"), env.get("_1130"), {"label":"/model/layers.4/post_attention_layernorm/SkipLayerNorm_mul_796"}));
  env.set("_1140", builder["matmul"](env.get("_1132"), env.get("_1139"), {"label":"/model/layers.4/mlp/gate_proj/MatMul_Q4_matmul_801"}));
  env.set("_1141", builder["sigmoid"](env.get("_1140"), {"label":"/model/layers.4/mlp/act_fn/Sigmoid_802"}));
  env.set("_1142", builder["mul"](env.get("_1140"), env.get("_1141"), {"label":"/model/layers.4/mlp/act_fn/Mul_803"}));
  env.set("_468", builder.dequantizeLinear(env.get("_465"), env.get("_466"), env.get("_467"), {"axis":2,"blockSize":32,"label":"/model/layers.4/mlp/up_proj/MatMul_Q4_dequantizeLinear_210"}));
  env.set("_469", builder.reshape(env.get("_468"), [5632,2048]));
  env.set("_470", builder["transpose"](env.get("_469"), {"label":"/model/layers.4/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_212","permutation":[1,0]}));
  env.set("_1133", builder["matmul"](env.get("_1132"), env.get("_470"), {"label":"/model/layers.4/mlp/up_proj/MatMul_Q4_matmul_797"}));
  env.set("_1143", builder["mul"](env.get("_1142"), env.get("_1133"), {"label":"/model/layers.4/mlp/Mul_804"}));
  env.set("_462", builder.dequantizeLinear(env.get("_459"), env.get("_460"), env.get("_461"), {"axis":2,"blockSize":32,"label":"/model/layers.4/mlp/down_proj/MatMul_Q4_dequantizeLinear_207"}));
  env.set("_463", builder.reshape(env.get("_462"), [2048,5632]));
  env.set("_464", builder["transpose"](env.get("_463"), {"label":"/model/layers.4/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_209","permutation":[1,0]}));
  env.set("_1144", builder["matmul"](env.get("_1143"), env.get("_464"), {"label":"/model/layers.4/mlp/down_proj/MatMul_Q4_matmul_805"}));
  env.set("_1145", builder["add"](env.get("_1125"), env.get("_1144"), {"label":"/model/layers.5/input_layernorm/SkipLayerNorm_add_skip_806"}));
  env.set("_1146", builder["pow"](env.get("_1145"), env.get("_582"), {"label":"/model/layers.5/input_layernorm/SkipLayerNorm_pow_807"}));
  env.set("_1147", builder["reduceMean"](env.get("_1146"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.5/input_layernorm/SkipLayerNorm_reduceMean_808"}));
  env.set("_1148", builder["add"](env.get("_1147"), env.get("_585"), {"label":"/model/layers.5/input_layernorm/SkipLayerNorm_add_809"}));
  env.set("_1149", builder["sqrt"](env.get("_1148"), {"label":"/model/layers.5/input_layernorm/SkipLayerNorm_sqrt_810"}));
  env.set("_1150", builder["div"](env.get("_1145"), env.get("_1149"), {"label":"/model/layers.5/input_layernorm/SkipLayerNorm_div_811"}));
  env.set("_1152", builder["mul"](env.get("_1151"), env.get("_1150"), {"label":"/model/layers.5/input_layernorm/SkipLayerNorm_mul_812"}));
  env.set("_1208", builder["matmul"](env.get("_1152"), env.get("_1207"), {"label":"/model/layers.5/attn/q_proj/MatMul_Q4_matmul_872"}));
  env.set("_1209", builder.reshape(env.get("_1208"), [1,sequence_length,32,64]));
  env.set("_1210", builder.reshape(env.get("_1209"), [1,sequence_length,32,2,32]));
  env.set("_1220", builder["mul"](env.get("_1210"), env.get("_1219"), {"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_mul_cos_883"}));
  env.set("_1221", builder.reshape(env.get("_1220"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_1210"), 2, {"axis":3,"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_split_partial_input0_875"});
    env.set("_1211", tmp[0]);
    env.set("_1212", tmp[1]);
  }
  env.set("_1213", builder.concat([env.get("_1212"), env.get("_1211")], 3, {"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_concat_partial_input0_876"}));
  env.set("Inserted_435", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_867","maxValue":2047,"minValue":-2048}));
  env.set("_1200", builder["gather"](env.get("_621"), env.get("Inserted_435"), {"axis":0,"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_gather_sin_866"}));
  env.set("_1201", builder.reshape(env.get("_1200"), [1,sequence_length,1,1,32]));
  env.set("_1214", builder["mul"](env.get("_1213"), env.get("_1201"), {"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_mul_sin_877"}));
  env.set("_1216", builder["mul"](env.get("_1214"), env.get("_1215"), {"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_mul_sign_878"}));
  env.set("_1217", builder.reshape(env.get("_1216"), [1,sequence_length,32,64]));
  env.set("_1222", builder["add"](env.get("_1221"), env.get("_1217"), {"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_885"}));
  env.set("_1223", builder.reshape(env.get("_1222"), [1,sequence_length,2048]));
  env.set("_1224", builder.reshape(env.get("_1223"), [1,sequence_length,32,64]));
  env.set("_1225", builder["transpose"](env.get("_1224"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/query/transpose_888","permutation":[0,2,1,3]}));
  env.set("Inserted_396", builder.cast(env.get("_598"), "uint8"));
  env.set("_1155", builder["where"](env.get("Inserted_396"), env.get("_599"), env.get("_597"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/scatter/where_815"}));
  env.set("_1156", builder["add"](env.get("_601"), env.get("_1155"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/right_constant/add_817"}));
  env.set("_1157", builder.concat([env.get("_603"), env.get("_1156")], 2, {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_818"}));
  env.set("_1158", builder.reshape(env.get("_1157"), [1,sequence_length,4,3]));
  env.set("Inserted_427", builder.cast(env.get("_1158"), "int64"));
  env.set("Inserted_429", builder["max"](env.get("Inserted_427"), env.get("Inserted_428"), {"label":"Inserted_Max_860"}));
  env.set("Inserted_431", builder["min"](env.get("Inserted_429"), env.get("Inserted_430"), {"label":"Inserted_Min_861"}));
  env.set("Inserted_420", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_851","maxValue":2047,"minValue":-2048}));
  env.set("_1188", builder["gather"](env.get("_641"), env.get("Inserted_420"), {"axis":0,"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_gather_cos_850"}));
  env.set("_1189", builder.reshape(env.get("_1188"), [1,sequence_length,1,1,32]));
  env.set("_1175", builder.dequantizeLinear(env.get("_1172"), env.get("_1173"), env.get("_1174"), {"axis":2,"blockSize":32,"label":"/model/layers.5/attn/k_proj/MatMul_Q4_dequantizeLinear_839"}));
  env.set("_1176", builder.reshape(env.get("_1175"), [256,2048]));
  env.set("_1177", builder["transpose"](env.get("_1176"), {"label":"/model/layers.5/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_841","permutation":[1,0]}));
  env.set("_1178", builder["matmul"](env.get("_1152"), env.get("_1177"), {"label":"/model/layers.5/attn/k_proj/MatMul_Q4_matmul_842"}));
  env.set("_1179", builder.reshape(env.get("_1178"), [1,sequence_length,4,64]));
  env.set("_1180", builder.reshape(env.get("_1179"), [1,sequence_length,4,2,32]));
  env.set("_1190", builder["mul"](env.get("_1180"), env.get("_1189"), {"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_mul_cos_853"}));
  env.set("_1191", builder.reshape(env.get("_1190"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_1180"), 2, {"axis":3,"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_split_partial_input0_845"});
    env.set("_1181", tmp[0]);
    env.set("_1182", tmp[1]);
  }
  env.set("_1183", builder.concat([env.get("_1182"), env.get("_1181")], 3, {"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_concat_partial_input0_846"}));
  env.set("Inserted_411", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_837","maxValue":2047,"minValue":-2048}));
  env.set("_1170", builder["gather"](env.get("_621"), env.get("Inserted_411"), {"axis":0,"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_gather_sin_836"}));
  env.set("_1171", builder.reshape(env.get("_1170"), [1,sequence_length,1,1,32]));
  env.set("_1184", builder["mul"](env.get("_1183"), env.get("_1171"), {"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_mul_sin_847"}));
  env.set("_1186", builder["mul"](env.get("_1184"), env.get("_1185"), {"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_mul_sign_848"}));
  env.set("_1187", builder.reshape(env.get("_1186"), [1,sequence_length,4,64]));
  env.set("_1192", builder["add"](env.get("_1191"), env.get("_1187"), {"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_855"}));
  env.set("_1193", builder.reshape(env.get("_1192"), [1,sequence_length,256]));
  env.set("_1194", builder.reshape(env.get("_1193"), [1,sequence_length,4,64]));
  env.set("present_5_key_10", builder["scatterND"](env.get("past_key_values_5_key_1195"), env.get("Inserted_431"), env.get("_1194"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/present_key/ScatterND_858"}));
  env.set("_1196", builder.reshape(env.get("present_5_key_10"), [1,4,1,past_sequence_length,64]));
  env.set("_1197", builder.expand(env.get("_1196"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/true_present_key/expand_863"}));
  env.set("_1198", builder.reshape(env.get("_1197"), [1,32,past_sequence_length,64]));
  env.set("_1199", builder["transpose"](env.get("_1198"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/present_key/transpose_865","permutation":[0,1,3,2]}));
  env.set("_1226", builder["matmul"](env.get("_1225"), env.get("_1199"), {"label":"/model/layers.5/attn/GroupQueryAttention_/Attention/qkv/matmul_1_889"}));
  env.set("_1227", builder["mul"](env.get("_1226"), env.get("_681"), {"label":"/model/layers.5/attn/GroupQueryAttention_/Attention/qkv/div_890"}));
  env.set("_1166", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_830"}));
  env.set("_1167", builder.cumulativeSum(env.get("_1166"), 3, {"exclusive":true,"label":"/model/layers.5/attn/GroupQueryAttention_range_of_mask_shape_831"}));
  env.set("_1163", builder["add"](env.get("_610"), env.get("_1155"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/attn_mask/add_827"}));
  env.set("_1164", builder.expand(env.get("_1163"), [past_sequence_length,sequence_length], {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/expand_neq_right_828"}));
  env.set("_1165", builder["transpose"](env.get("_1164"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/neq_right/transpose_829","permutation":[1,0]}));
  env.set("Inserted_409", builder["lesser"](env.get("_1167"), env.get("_1165"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_832"}));
  env.set("_1168", builder.cast(env.get("Inserted_409"), "uint8"));
  env.set("Inserted_410", builder.cast(env.get("_1168"), "uint8"));
  env.set("_1169", builder["where"](env.get("Inserted_410"), env.get("_618"), env.get("_619"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/attn_mask/where_834"}));
  env.set("_1228", builder["add"](env.get("_1227"), env.get("_1169"), {"label":"/model/layers.5/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_891"}));
  env.set("_1229", builder["softmax"](env.get("_1228"), 3));
  env.set("Inserted_398", builder.cast(env.get("_1158"), "int64"));
  env.set("Inserted_400", builder["max"](env.get("Inserted_398"), env.get("Inserted_399"), {"label":"Inserted_Max_822"}));
  env.set("Inserted_402", builder["min"](env.get("Inserted_400"), env.get("Inserted_401"), {"label":"Inserted_Min_823"}));
  env.set("_456", builder.dequantizeLinear(env.get("_453"), env.get("_454"), env.get("_455"), {"axis":2,"blockSize":32,"label":"/model/layers.5/attn/v_proj/MatMul_Q4_dequantizeLinear_204"}));
  env.set("_457", builder.reshape(env.get("_456"), [256,2048]));
  env.set("_458", builder["transpose"](env.get("_457"), {"label":"/model/layers.5/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_206","permutation":[1,0]}));
  env.set("_1153", builder["matmul"](env.get("_1152"), env.get("_458"), {"label":"/model/layers.5/attn/v_proj/MatMul_Q4_matmul_813"}));
  env.set("_1154", builder.reshape(env.get("_1153"), [1,sequence_length,4,64]));
  env.set("present_5_value_11", builder["scatterND"](env.get("past_key_values_5_value_1159"), env.get("Inserted_402"), env.get("_1154"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/present_value/ScatterND_820"}));
  env.set("_1160", builder.reshape(env.get("present_5_value_11"), [1,4,1,past_sequence_length,64]));
  env.set("_1161", builder.expand(env.get("_1160"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/true_present_value/expand_825"}));
  env.set("_1162", builder.reshape(env.get("_1161"), [1,32,past_sequence_length,64]));
  env.set("_1230", builder["matmul"](env.get("_1229"), env.get("_1162"), {"label":"/model/layers.5/attn/GroupQueryAttention_/Attention/qkv/matmul_2_893"}));
  env.set("_1231", builder["transpose"](env.get("_1230"), {"label":"/model/layers.5/attn/GroupQueryAttention_/Attention/qkv/transpose_894","permutation":[0,2,1,3]}));
  env.set("_1232", builder.reshape(env.get("_1231"), [1,sequence_length,2048]));
  env.set("_450", builder.dequantizeLinear(env.get("_447"), env.get("_448"), env.get("_449"), {"axis":2,"blockSize":32,"label":"/model/layers.5/attn/o_proj/MatMul_Q4_dequantizeLinear_201"}));
  env.set("_451", builder.reshape(env.get("_450"), [2048,2048]));
  env.set("_452", builder["transpose"](env.get("_451"), {"label":"/model/layers.5/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_203","permutation":[1,0]}));
  env.set("_1233", builder["matmul"](env.get("_1232"), env.get("_452"), {"label":"/model/layers.5/attn/o_proj/MatMul_Q4_matmul_896"}));
  env.set("_1234", builder["add"](env.get("_1145"), env.get("_1233"), {"label":"/model/layers.5/post_attention_layernorm/SkipLayerNorm_add_skip_897"}));
  env.set("_1235", builder["pow"](env.get("_1234"), env.get("_582"), {"label":"/model/layers.5/post_attention_layernorm/SkipLayerNorm_pow_898"}));
  env.set("_1236", builder["reduceMean"](env.get("_1235"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.5/post_attention_layernorm/SkipLayerNorm_reduceMean_899"}));
  env.set("_1237", builder["add"](env.get("_1236"), env.get("_585"), {"label":"/model/layers.5/post_attention_layernorm/SkipLayerNorm_add_900"}));
  env.set("_1238", builder["sqrt"](env.get("_1237"), {"label":"/model/layers.5/post_attention_layernorm/SkipLayerNorm_sqrt_901"}));
  env.set("_1239", builder["div"](env.get("_1234"), env.get("_1238"), {"label":"/model/layers.5/post_attention_layernorm/SkipLayerNorm_div_902"}));
  env.set("_1241", builder["mul"](env.get("_1240"), env.get("_1239"), {"label":"/model/layers.5/post_attention_layernorm/SkipLayerNorm_mul_903"}));
  env.set("_1249", builder["matmul"](env.get("_1241"), env.get("_1248"), {"label":"/model/layers.5/mlp/gate_proj/MatMul_Q4_matmul_908"}));
  env.set("_1250", builder["sigmoid"](env.get("_1249"), {"label":"/model/layers.5/mlp/act_fn/Sigmoid_909"}));
  env.set("_1251", builder["mul"](env.get("_1249"), env.get("_1250"), {"label":"/model/layers.5/mlp/act_fn/Mul_910"}));
  env.set("_444", builder.dequantizeLinear(env.get("_441"), env.get("_442"), env.get("_443"), {"axis":2,"blockSize":32,"label":"/model/layers.5/mlp/up_proj/MatMul_Q4_dequantizeLinear_198"}));
  env.set("_445", builder.reshape(env.get("_444"), [5632,2048]));
  env.set("_446", builder["transpose"](env.get("_445"), {"label":"/model/layers.5/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_200","permutation":[1,0]}));
  env.set("_1242", builder["matmul"](env.get("_1241"), env.get("_446"), {"label":"/model/layers.5/mlp/up_proj/MatMul_Q4_matmul_904"}));
  env.set("_1252", builder["mul"](env.get("_1251"), env.get("_1242"), {"label":"/model/layers.5/mlp/Mul_911"}));
  env.set("_438", builder.dequantizeLinear(env.get("_435"), env.get("_436"), env.get("_437"), {"axis":2,"blockSize":32,"label":"/model/layers.5/mlp/down_proj/MatMul_Q4_dequantizeLinear_195"}));
  env.set("_439", builder.reshape(env.get("_438"), [2048,5632]));
  env.set("_440", builder["transpose"](env.get("_439"), {"label":"/model/layers.5/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_197","permutation":[1,0]}));
  env.set("_1253", builder["matmul"](env.get("_1252"), env.get("_440"), {"label":"/model/layers.5/mlp/down_proj/MatMul_Q4_matmul_912"}));
  env.set("_1254", builder["add"](env.get("_1234"), env.get("_1253"), {"label":"/model/layers.6/input_layernorm/SkipLayerNorm_add_skip_913"}));
  env.set("_1255", builder["pow"](env.get("_1254"), env.get("_582"), {"label":"/model/layers.6/input_layernorm/SkipLayerNorm_pow_914"}));
  env.set("_1256", builder["reduceMean"](env.get("_1255"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.6/input_layernorm/SkipLayerNorm_reduceMean_915"}));
  env.set("_1257", builder["add"](env.get("_1256"), env.get("_585"), {"label":"/model/layers.6/input_layernorm/SkipLayerNorm_add_916"}));
  env.set("_1258", builder["sqrt"](env.get("_1257"), {"label":"/model/layers.6/input_layernorm/SkipLayerNorm_sqrt_917"}));
  env.set("_1259", builder["div"](env.get("_1254"), env.get("_1258"), {"label":"/model/layers.6/input_layernorm/SkipLayerNorm_div_918"}));
  env.set("_1261", builder["mul"](env.get("_1260"), env.get("_1259"), {"label":"/model/layers.6/input_layernorm/SkipLayerNorm_mul_919"}));
  env.set("_1317", builder["matmul"](env.get("_1261"), env.get("_1316"), {"label":"/model/layers.6/attn/q_proj/MatMul_Q4_matmul_979"}));
  env.set("_1318", builder.reshape(env.get("_1317"), [1,sequence_length,32,64]));
  env.set("_1319", builder.reshape(env.get("_1318"), [1,sequence_length,32,2,32]));
  env.set("_1329", builder["mul"](env.get("_1319"), env.get("_1328"), {"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_mul_cos_990"}));
  env.set("_1330", builder.reshape(env.get("_1329"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_1319"), 2, {"axis":3,"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_split_partial_input0_982"});
    env.set("_1320", tmp[0]);
    env.set("_1321", tmp[1]);
  }
  env.set("_1322", builder.concat([env.get("_1321"), env.get("_1320")], 3, {"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_concat_partial_input0_983"}));
  env.set("Inserted_495", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_974","maxValue":2047,"minValue":-2048}));
  env.set("_1309", builder["gather"](env.get("_621"), env.get("Inserted_495"), {"axis":0,"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_gather_sin_973"}));
  env.set("_1310", builder.reshape(env.get("_1309"), [1,sequence_length,1,1,32]));
  env.set("_1323", builder["mul"](env.get("_1322"), env.get("_1310"), {"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_mul_sin_984"}));
  env.set("_1325", builder["mul"](env.get("_1323"), env.get("_1324"), {"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_mul_sign_985"}));
  env.set("_1326", builder.reshape(env.get("_1325"), [1,sequence_length,32,64]));
  env.set("_1331", builder["add"](env.get("_1330"), env.get("_1326"), {"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_992"}));
  env.set("_1332", builder.reshape(env.get("_1331"), [1,sequence_length,2048]));
  env.set("_1333", builder.reshape(env.get("_1332"), [1,sequence_length,32,64]));
  env.set("_1334", builder["transpose"](env.get("_1333"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/query/transpose_995","permutation":[0,2,1,3]}));
  env.set("Inserted_456", builder.cast(env.get("_598"), "uint8"));
  env.set("_1264", builder["where"](env.get("Inserted_456"), env.get("_599"), env.get("_597"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/scatter/where_922"}));
  env.set("_1265", builder["add"](env.get("_601"), env.get("_1264"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/right_constant/add_924"}));
  env.set("_1266", builder.concat([env.get("_603"), env.get("_1265")], 2, {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_925"}));
  env.set("_1267", builder.reshape(env.get("_1266"), [1,sequence_length,4,3]));
  env.set("Inserted_487", builder.cast(env.get("_1267"), "int64"));
  env.set("Inserted_489", builder["max"](env.get("Inserted_487"), env.get("Inserted_488"), {"label":"Inserted_Max_967"}));
  env.set("Inserted_491", builder["min"](env.get("Inserted_489"), env.get("Inserted_490"), {"label":"Inserted_Min_968"}));
  env.set("Inserted_480", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_958","maxValue":2047,"minValue":-2048}));
  env.set("_1297", builder["gather"](env.get("_641"), env.get("Inserted_480"), {"axis":0,"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_gather_cos_957"}));
  env.set("_1298", builder.reshape(env.get("_1297"), [1,sequence_length,1,1,32]));
  env.set("_1284", builder.dequantizeLinear(env.get("_1281"), env.get("_1282"), env.get("_1283"), {"axis":2,"blockSize":32,"label":"/model/layers.6/attn/k_proj/MatMul_Q4_dequantizeLinear_946"}));
  env.set("_1285", builder.reshape(env.get("_1284"), [256,2048]));
  env.set("_1286", builder["transpose"](env.get("_1285"), {"label":"/model/layers.6/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_948","permutation":[1,0]}));
  env.set("_1287", builder["matmul"](env.get("_1261"), env.get("_1286"), {"label":"/model/layers.6/attn/k_proj/MatMul_Q4_matmul_949"}));
  env.set("_1288", builder.reshape(env.get("_1287"), [1,sequence_length,4,64]));
  env.set("_1289", builder.reshape(env.get("_1288"), [1,sequence_length,4,2,32]));
  env.set("_1299", builder["mul"](env.get("_1289"), env.get("_1298"), {"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_mul_cos_960"}));
  env.set("_1300", builder.reshape(env.get("_1299"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_1289"), 2, {"axis":3,"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_split_partial_input0_952"});
    env.set("_1290", tmp[0]);
    env.set("_1291", tmp[1]);
  }
  env.set("_1292", builder.concat([env.get("_1291"), env.get("_1290")], 3, {"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_concat_partial_input0_953"}));
  env.set("Inserted_471", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_944","maxValue":2047,"minValue":-2048}));
  env.set("_1279", builder["gather"](env.get("_621"), env.get("Inserted_471"), {"axis":0,"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_gather_sin_943"}));
  env.set("_1280", builder.reshape(env.get("_1279"), [1,sequence_length,1,1,32]));
  env.set("_1293", builder["mul"](env.get("_1292"), env.get("_1280"), {"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_mul_sin_954"}));
  env.set("_1295", builder["mul"](env.get("_1293"), env.get("_1294"), {"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_mul_sign_955"}));
  env.set("_1296", builder.reshape(env.get("_1295"), [1,sequence_length,4,64]));
  env.set("_1301", builder["add"](env.get("_1300"), env.get("_1296"), {"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_962"}));
  env.set("_1302", builder.reshape(env.get("_1301"), [1,sequence_length,256]));
  env.set("_1303", builder.reshape(env.get("_1302"), [1,sequence_length,4,64]));
  env.set("present_6_key_12", builder["scatterND"](env.get("past_key_values_6_key_1304"), env.get("Inserted_491"), env.get("_1303"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/present_key/ScatterND_965"}));
  env.set("_1305", builder.reshape(env.get("present_6_key_12"), [1,4,1,past_sequence_length,64]));
  env.set("_1306", builder.expand(env.get("_1305"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/true_present_key/expand_970"}));
  env.set("_1307", builder.reshape(env.get("_1306"), [1,32,past_sequence_length,64]));
  env.set("_1308", builder["transpose"](env.get("_1307"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/present_key/transpose_972","permutation":[0,1,3,2]}));
  env.set("_1335", builder["matmul"](env.get("_1334"), env.get("_1308"), {"label":"/model/layers.6/attn/GroupQueryAttention_/Attention/qkv/matmul_1_996"}));
  env.set("_1336", builder["mul"](env.get("_1335"), env.get("_681"), {"label":"/model/layers.6/attn/GroupQueryAttention_/Attention/qkv/div_997"}));
  env.set("_1275", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_937"}));
  env.set("_1276", builder.cumulativeSum(env.get("_1275"), 3, {"exclusive":true,"label":"/model/layers.6/attn/GroupQueryAttention_range_of_mask_shape_938"}));
  env.set("_1272", builder["add"](env.get("_610"), env.get("_1264"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/attn_mask/add_934"}));
  env.set("_1273", builder.expand(env.get("_1272"), [past_sequence_length,sequence_length], {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/expand_neq_right_935"}));
  env.set("_1274", builder["transpose"](env.get("_1273"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/neq_right/transpose_936","permutation":[1,0]}));
  env.set("Inserted_469", builder["lesser"](env.get("_1276"), env.get("_1274"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_939"}));
  env.set("_1277", builder.cast(env.get("Inserted_469"), "uint8"));
  env.set("Inserted_470", builder.cast(env.get("_1277"), "uint8"));
  env.set("_1278", builder["where"](env.get("Inserted_470"), env.get("_618"), env.get("_619"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/attn_mask/where_941"}));
  env.set("_1337", builder["add"](env.get("_1336"), env.get("_1278"), {"label":"/model/layers.6/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_998"}));
  env.set("_1338", builder["softmax"](env.get("_1337"), 3));
  env.set("Inserted_458", builder.cast(env.get("_1267"), "int64"));
  env.set("Inserted_460", builder["max"](env.get("Inserted_458"), env.get("Inserted_459"), {"label":"Inserted_Max_929"}));
  env.set("Inserted_462", builder["min"](env.get("Inserted_460"), env.get("Inserted_461"), {"label":"Inserted_Min_930"}));
  env.set("_432", builder.dequantizeLinear(env.get("_429"), env.get("_430"), env.get("_431"), {"axis":2,"blockSize":32,"label":"/model/layers.6/attn/v_proj/MatMul_Q4_dequantizeLinear_192"}));
  env.set("_433", builder.reshape(env.get("_432"), [256,2048]));
  env.set("_434", builder["transpose"](env.get("_433"), {"label":"/model/layers.6/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_194","permutation":[1,0]}));
  env.set("_1262", builder["matmul"](env.get("_1261"), env.get("_434"), {"label":"/model/layers.6/attn/v_proj/MatMul_Q4_matmul_920"}));
  env.set("_1263", builder.reshape(env.get("_1262"), [1,sequence_length,4,64]));
  env.set("present_6_value_13", builder["scatterND"](env.get("past_key_values_6_value_1268"), env.get("Inserted_462"), env.get("_1263"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/present_value/ScatterND_927"}));
  env.set("_1269", builder.reshape(env.get("present_6_value_13"), [1,4,1,past_sequence_length,64]));
  env.set("_1270", builder.expand(env.get("_1269"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/true_present_value/expand_932"}));
  env.set("_1271", builder.reshape(env.get("_1270"), [1,32,past_sequence_length,64]));
  env.set("_1339", builder["matmul"](env.get("_1338"), env.get("_1271"), {"label":"/model/layers.6/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1000"}));
  env.set("_1340", builder["transpose"](env.get("_1339"), {"label":"/model/layers.6/attn/GroupQueryAttention_/Attention/qkv/transpose_1001","permutation":[0,2,1,3]}));
  env.set("_1341", builder.reshape(env.get("_1340"), [1,sequence_length,2048]));
  env.set("_426", builder.dequantizeLinear(env.get("_423"), env.get("_424"), env.get("_425"), {"axis":2,"blockSize":32,"label":"/model/layers.6/attn/o_proj/MatMul_Q4_dequantizeLinear_189"}));
  env.set("_427", builder.reshape(env.get("_426"), [2048,2048]));
  env.set("_428", builder["transpose"](env.get("_427"), {"label":"/model/layers.6/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_191","permutation":[1,0]}));
  env.set("_1342", builder["matmul"](env.get("_1341"), env.get("_428"), {"label":"/model/layers.6/attn/o_proj/MatMul_Q4_matmul_1003"}));
  env.set("_1343", builder["add"](env.get("_1254"), env.get("_1342"), {"label":"/model/layers.6/post_attention_layernorm/SkipLayerNorm_add_skip_1004"}));
  env.set("_1344", builder["pow"](env.get("_1343"), env.get("_582"), {"label":"/model/layers.6/post_attention_layernorm/SkipLayerNorm_pow_1005"}));
  env.set("_1345", builder["reduceMean"](env.get("_1344"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.6/post_attention_layernorm/SkipLayerNorm_reduceMean_1006"}));
  env.set("_1346", builder["add"](env.get("_1345"), env.get("_585"), {"label":"/model/layers.6/post_attention_layernorm/SkipLayerNorm_add_1007"}));
  env.set("_1347", builder["sqrt"](env.get("_1346"), {"label":"/model/layers.6/post_attention_layernorm/SkipLayerNorm_sqrt_1008"}));
  env.set("_1348", builder["div"](env.get("_1343"), env.get("_1347"), {"label":"/model/layers.6/post_attention_layernorm/SkipLayerNorm_div_1009"}));
  env.set("_1350", builder["mul"](env.get("_1349"), env.get("_1348"), {"label":"/model/layers.6/post_attention_layernorm/SkipLayerNorm_mul_1010"}));
  env.set("_1358", builder["matmul"](env.get("_1350"), env.get("_1357"), {"label":"/model/layers.6/mlp/gate_proj/MatMul_Q4_matmul_1015"}));
  env.set("_1359", builder["sigmoid"](env.get("_1358"), {"label":"/model/layers.6/mlp/act_fn/Sigmoid_1016"}));
  env.set("_1360", builder["mul"](env.get("_1358"), env.get("_1359"), {"label":"/model/layers.6/mlp/act_fn/Mul_1017"}));
  env.set("_420", builder.dequantizeLinear(env.get("_417"), env.get("_418"), env.get("_419"), {"axis":2,"blockSize":32,"label":"/model/layers.6/mlp/up_proj/MatMul_Q4_dequantizeLinear_186"}));
  env.set("_421", builder.reshape(env.get("_420"), [5632,2048]));
  env.set("_422", builder["transpose"](env.get("_421"), {"label":"/model/layers.6/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_188","permutation":[1,0]}));
  env.set("_1351", builder["matmul"](env.get("_1350"), env.get("_422"), {"label":"/model/layers.6/mlp/up_proj/MatMul_Q4_matmul_1011"}));
  env.set("_1361", builder["mul"](env.get("_1360"), env.get("_1351"), {"label":"/model/layers.6/mlp/Mul_1018"}));
  env.set("_414", builder.dequantizeLinear(env.get("_411"), env.get("_412"), env.get("_413"), {"axis":2,"blockSize":32,"label":"/model/layers.6/mlp/down_proj/MatMul_Q4_dequantizeLinear_183"}));
  env.set("_415", builder.reshape(env.get("_414"), [2048,5632]));
  env.set("_416", builder["transpose"](env.get("_415"), {"label":"/model/layers.6/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_185","permutation":[1,0]}));
  env.set("_1362", builder["matmul"](env.get("_1361"), env.get("_416"), {"label":"/model/layers.6/mlp/down_proj/MatMul_Q4_matmul_1019"}));
  env.set("_1363", builder["add"](env.get("_1343"), env.get("_1362"), {"label":"/model/layers.7/input_layernorm/SkipLayerNorm_add_skip_1020"}));
  env.set("_1364", builder["pow"](env.get("_1363"), env.get("_582"), {"label":"/model/layers.7/input_layernorm/SkipLayerNorm_pow_1021"}));
  env.set("_1365", builder["reduceMean"](env.get("_1364"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.7/input_layernorm/SkipLayerNorm_reduceMean_1022"}));
  env.set("_1366", builder["add"](env.get("_1365"), env.get("_585"), {"label":"/model/layers.7/input_layernorm/SkipLayerNorm_add_1023"}));
  env.set("_1367", builder["sqrt"](env.get("_1366"), {"label":"/model/layers.7/input_layernorm/SkipLayerNorm_sqrt_1024"}));
  env.set("_1368", builder["div"](env.get("_1363"), env.get("_1367"), {"label":"/model/layers.7/input_layernorm/SkipLayerNorm_div_1025"}));
  env.set("_1370", builder["mul"](env.get("_1369"), env.get("_1368"), {"label":"/model/layers.7/input_layernorm/SkipLayerNorm_mul_1026"}));
  env.set("_1426", builder["matmul"](env.get("_1370"), env.get("_1425"), {"label":"/model/layers.7/attn/q_proj/MatMul_Q4_matmul_1086"}));
  env.set("_1427", builder.reshape(env.get("_1426"), [1,sequence_length,32,64]));
  env.set("_1428", builder.reshape(env.get("_1427"), [1,sequence_length,32,2,32]));
  env.set("_1438", builder["mul"](env.get("_1428"), env.get("_1437"), {"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_mul_cos_1097"}));
  env.set("_1439", builder.reshape(env.get("_1438"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_1428"), 2, {"axis":3,"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_split_partial_input0_1089"});
    env.set("_1429", tmp[0]);
    env.set("_1430", tmp[1]);
  }
  env.set("_1431", builder.concat([env.get("_1430"), env.get("_1429")], 3, {"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1090"}));
  env.set("Inserted_555", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1081","maxValue":2047,"minValue":-2048}));
  env.set("_1418", builder["gather"](env.get("_621"), env.get("Inserted_555"), {"axis":0,"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_gather_sin_1080"}));
  env.set("_1419", builder.reshape(env.get("_1418"), [1,sequence_length,1,1,32]));
  env.set("_1432", builder["mul"](env.get("_1431"), env.get("_1419"), {"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_mul_sin_1091"}));
  env.set("_1434", builder["mul"](env.get("_1432"), env.get("_1433"), {"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_mul_sign_1092"}));
  env.set("_1435", builder.reshape(env.get("_1434"), [1,sequence_length,32,64]));
  env.set("_1440", builder["add"](env.get("_1439"), env.get("_1435"), {"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1099"}));
  env.set("_1441", builder.reshape(env.get("_1440"), [1,sequence_length,2048]));
  env.set("_1442", builder.reshape(env.get("_1441"), [1,sequence_length,32,64]));
  env.set("_1443", builder["transpose"](env.get("_1442"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/query/transpose_1102","permutation":[0,2,1,3]}));
  env.set("Inserted_516", builder.cast(env.get("_598"), "uint8"));
  env.set("_1373", builder["where"](env.get("Inserted_516"), env.get("_599"), env.get("_597"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/scatter/where_1029"}));
  env.set("_1374", builder["add"](env.get("_601"), env.get("_1373"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/right_constant/add_1031"}));
  env.set("_1375", builder.concat([env.get("_603"), env.get("_1374")], 2, {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1032"}));
  env.set("_1376", builder.reshape(env.get("_1375"), [1,sequence_length,4,3]));
  env.set("Inserted_547", builder.cast(env.get("_1376"), "int64"));
  env.set("Inserted_549", builder["max"](env.get("Inserted_547"), env.get("Inserted_548"), {"label":"Inserted_Max_1074"}));
  env.set("Inserted_551", builder["min"](env.get("Inserted_549"), env.get("Inserted_550"), {"label":"Inserted_Min_1075"}));
  env.set("Inserted_540", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1065","maxValue":2047,"minValue":-2048}));
  env.set("_1406", builder["gather"](env.get("_641"), env.get("Inserted_540"), {"axis":0,"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_gather_cos_1064"}));
  env.set("_1407", builder.reshape(env.get("_1406"), [1,sequence_length,1,1,32]));
  env.set("_1393", builder.dequantizeLinear(env.get("_1390"), env.get("_1391"), env.get("_1392"), {"axis":2,"blockSize":32,"label":"/model/layers.7/attn/k_proj/MatMul_Q4_dequantizeLinear_1053"}));
  env.set("_1394", builder.reshape(env.get("_1393"), [256,2048]));
  env.set("_1395", builder["transpose"](env.get("_1394"), {"label":"/model/layers.7/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1055","permutation":[1,0]}));
  env.set("_1396", builder["matmul"](env.get("_1370"), env.get("_1395"), {"label":"/model/layers.7/attn/k_proj/MatMul_Q4_matmul_1056"}));
  env.set("_1397", builder.reshape(env.get("_1396"), [1,sequence_length,4,64]));
  env.set("_1398", builder.reshape(env.get("_1397"), [1,sequence_length,4,2,32]));
  env.set("_1408", builder["mul"](env.get("_1398"), env.get("_1407"), {"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_mul_cos_1067"}));
  env.set("_1409", builder.reshape(env.get("_1408"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_1398"), 2, {"axis":3,"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_split_partial_input0_1059"});
    env.set("_1399", tmp[0]);
    env.set("_1400", tmp[1]);
  }
  env.set("_1401", builder.concat([env.get("_1400"), env.get("_1399")], 3, {"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1060"}));
  env.set("Inserted_531", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1051","maxValue":2047,"minValue":-2048}));
  env.set("_1388", builder["gather"](env.get("_621"), env.get("Inserted_531"), {"axis":0,"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_gather_sin_1050"}));
  env.set("_1389", builder.reshape(env.get("_1388"), [1,sequence_length,1,1,32]));
  env.set("_1402", builder["mul"](env.get("_1401"), env.get("_1389"), {"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_mul_sin_1061"}));
  env.set("_1404", builder["mul"](env.get("_1402"), env.get("_1403"), {"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_mul_sign_1062"}));
  env.set("_1405", builder.reshape(env.get("_1404"), [1,sequence_length,4,64]));
  env.set("_1410", builder["add"](env.get("_1409"), env.get("_1405"), {"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1069"}));
  env.set("_1411", builder.reshape(env.get("_1410"), [1,sequence_length,256]));
  env.set("_1412", builder.reshape(env.get("_1411"), [1,sequence_length,4,64]));
  env.set("present_7_key_14", builder["scatterND"](env.get("past_key_values_7_key_1413"), env.get("Inserted_551"), env.get("_1412"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1072"}));
  env.set("_1414", builder.reshape(env.get("present_7_key_14"), [1,4,1,past_sequence_length,64]));
  env.set("_1415", builder.expand(env.get("_1414"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/true_present_key/expand_1077"}));
  env.set("_1416", builder.reshape(env.get("_1415"), [1,32,past_sequence_length,64]));
  env.set("_1417", builder["transpose"](env.get("_1416"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/present_key/transpose_1079","permutation":[0,1,3,2]}));
  env.set("_1444", builder["matmul"](env.get("_1443"), env.get("_1417"), {"label":"/model/layers.7/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1103"}));
  env.set("_1445", builder["mul"](env.get("_1444"), env.get("_681"), {"label":"/model/layers.7/attn/GroupQueryAttention_/Attention/qkv/div_1104"}));
  env.set("_1384", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1044"}));
  env.set("_1385", builder.cumulativeSum(env.get("_1384"), 3, {"exclusive":true,"label":"/model/layers.7/attn/GroupQueryAttention_range_of_mask_shape_1045"}));
  env.set("_1381", builder["add"](env.get("_610"), env.get("_1373"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/attn_mask/add_1041"}));
  env.set("_1382", builder.expand(env.get("_1381"), [past_sequence_length,sequence_length], {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/expand_neq_right_1042"}));
  env.set("_1383", builder["transpose"](env.get("_1382"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/neq_right/transpose_1043","permutation":[1,0]}));
  env.set("Inserted_529", builder["lesser"](env.get("_1385"), env.get("_1383"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1046"}));
  env.set("_1386", builder.cast(env.get("Inserted_529"), "uint8"));
  env.set("Inserted_530", builder.cast(env.get("_1386"), "uint8"));
  env.set("_1387", builder["where"](env.get("Inserted_530"), env.get("_618"), env.get("_619"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/attn_mask/where_1048"}));
  env.set("_1446", builder["add"](env.get("_1445"), env.get("_1387"), {"label":"/model/layers.7/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1105"}));
  env.set("_1447", builder["softmax"](env.get("_1446"), 3));
  env.set("Inserted_518", builder.cast(env.get("_1376"), "int64"));
  env.set("Inserted_520", builder["max"](env.get("Inserted_518"), env.get("Inserted_519"), {"label":"Inserted_Max_1036"}));
  env.set("Inserted_522", builder["min"](env.get("Inserted_520"), env.get("Inserted_521"), {"label":"Inserted_Min_1037"}));
  env.set("_408", builder.dequantizeLinear(env.get("_405"), env.get("_406"), env.get("_407"), {"axis":2,"blockSize":32,"label":"/model/layers.7/attn/v_proj/MatMul_Q4_dequantizeLinear_180"}));
  env.set("_409", builder.reshape(env.get("_408"), [256,2048]));
  env.set("_410", builder["transpose"](env.get("_409"), {"label":"/model/layers.7/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_182","permutation":[1,0]}));
  env.set("_1371", builder["matmul"](env.get("_1370"), env.get("_410"), {"label":"/model/layers.7/attn/v_proj/MatMul_Q4_matmul_1027"}));
  env.set("_1372", builder.reshape(env.get("_1371"), [1,sequence_length,4,64]));
  env.set("present_7_value_15", builder["scatterND"](env.get("past_key_values_7_value_1377"), env.get("Inserted_522"), env.get("_1372"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1034"}));
  env.set("_1378", builder.reshape(env.get("present_7_value_15"), [1,4,1,past_sequence_length,64]));
  env.set("_1379", builder.expand(env.get("_1378"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/true_present_value/expand_1039"}));
  env.set("_1380", builder.reshape(env.get("_1379"), [1,32,past_sequence_length,64]));
  env.set("_1448", builder["matmul"](env.get("_1447"), env.get("_1380"), {"label":"/model/layers.7/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1107"}));
  env.set("_1449", builder["transpose"](env.get("_1448"), {"label":"/model/layers.7/attn/GroupQueryAttention_/Attention/qkv/transpose_1108","permutation":[0,2,1,3]}));
  env.set("_1450", builder.reshape(env.get("_1449"), [1,sequence_length,2048]));
  env.set("_402", builder.dequantizeLinear(env.get("_399"), env.get("_400"), env.get("_401"), {"axis":2,"blockSize":32,"label":"/model/layers.7/attn/o_proj/MatMul_Q4_dequantizeLinear_177"}));
  env.set("_403", builder.reshape(env.get("_402"), [2048,2048]));
  env.set("_404", builder["transpose"](env.get("_403"), {"label":"/model/layers.7/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_179","permutation":[1,0]}));
  env.set("_1451", builder["matmul"](env.get("_1450"), env.get("_404"), {"label":"/model/layers.7/attn/o_proj/MatMul_Q4_matmul_1110"}));
  env.set("_1452", builder["add"](env.get("_1363"), env.get("_1451"), {"label":"/model/layers.7/post_attention_layernorm/SkipLayerNorm_add_skip_1111"}));
  env.set("_1453", builder["pow"](env.get("_1452"), env.get("_582"), {"label":"/model/layers.7/post_attention_layernorm/SkipLayerNorm_pow_1112"}));
  env.set("_1454", builder["reduceMean"](env.get("_1453"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.7/post_attention_layernorm/SkipLayerNorm_reduceMean_1113"}));
  env.set("_1455", builder["add"](env.get("_1454"), env.get("_585"), {"label":"/model/layers.7/post_attention_layernorm/SkipLayerNorm_add_1114"}));
  env.set("_1456", builder["sqrt"](env.get("_1455"), {"label":"/model/layers.7/post_attention_layernorm/SkipLayerNorm_sqrt_1115"}));
  env.set("_1457", builder["div"](env.get("_1452"), env.get("_1456"), {"label":"/model/layers.7/post_attention_layernorm/SkipLayerNorm_div_1116"}));
  env.set("_1459", builder["mul"](env.get("_1458"), env.get("_1457"), {"label":"/model/layers.7/post_attention_layernorm/SkipLayerNorm_mul_1117"}));
  env.set("_1467", builder["matmul"](env.get("_1459"), env.get("_1466"), {"label":"/model/layers.7/mlp/gate_proj/MatMul_Q4_matmul_1122"}));
  env.set("_1468", builder["sigmoid"](env.get("_1467"), {"label":"/model/layers.7/mlp/act_fn/Sigmoid_1123"}));
  env.set("_1469", builder["mul"](env.get("_1467"), env.get("_1468"), {"label":"/model/layers.7/mlp/act_fn/Mul_1124"}));
  env.set("_396", builder.dequantizeLinear(env.get("_393"), env.get("_394"), env.get("_395"), {"axis":2,"blockSize":32,"label":"/model/layers.7/mlp/up_proj/MatMul_Q4_dequantizeLinear_174"}));
  env.set("_397", builder.reshape(env.get("_396"), [5632,2048]));
  env.set("_398", builder["transpose"](env.get("_397"), {"label":"/model/layers.7/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_176","permutation":[1,0]}));
  env.set("_1460", builder["matmul"](env.get("_1459"), env.get("_398"), {"label":"/model/layers.7/mlp/up_proj/MatMul_Q4_matmul_1118"}));
  env.set("_1470", builder["mul"](env.get("_1469"), env.get("_1460"), {"label":"/model/layers.7/mlp/Mul_1125"}));
  env.set("_390", builder.dequantizeLinear(env.get("_387"), env.get("_388"), env.get("_389"), {"axis":2,"blockSize":32,"label":"/model/layers.7/mlp/down_proj/MatMul_Q4_dequantizeLinear_171"}));
  env.set("_391", builder.reshape(env.get("_390"), [2048,5632]));
  env.set("_392", builder["transpose"](env.get("_391"), {"label":"/model/layers.7/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_173","permutation":[1,0]}));
  env.set("_1471", builder["matmul"](env.get("_1470"), env.get("_392"), {"label":"/model/layers.7/mlp/down_proj/MatMul_Q4_matmul_1126"}));
  env.set("_1472", builder["add"](env.get("_1452"), env.get("_1471"), {"label":"/model/layers.8/input_layernorm/SkipLayerNorm_add_skip_1127"}));
  env.set("_1473", builder["pow"](env.get("_1472"), env.get("_582"), {"label":"/model/layers.8/input_layernorm/SkipLayerNorm_pow_1128"}));
  env.set("_1474", builder["reduceMean"](env.get("_1473"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.8/input_layernorm/SkipLayerNorm_reduceMean_1129"}));
  env.set("_1475", builder["add"](env.get("_1474"), env.get("_585"), {"label":"/model/layers.8/input_layernorm/SkipLayerNorm_add_1130"}));
  env.set("_1476", builder["sqrt"](env.get("_1475"), {"label":"/model/layers.8/input_layernorm/SkipLayerNorm_sqrt_1131"}));
  env.set("_1477", builder["div"](env.get("_1472"), env.get("_1476"), {"label":"/model/layers.8/input_layernorm/SkipLayerNorm_div_1132"}));
  env.set("_1479", builder["mul"](env.get("_1478"), env.get("_1477"), {"label":"/model/layers.8/input_layernorm/SkipLayerNorm_mul_1133"}));
  env.set("_1535", builder["matmul"](env.get("_1479"), env.get("_1534"), {"label":"/model/layers.8/attn/q_proj/MatMul_Q4_matmul_1193"}));
  env.set("_1536", builder.reshape(env.get("_1535"), [1,sequence_length,32,64]));
  env.set("_1537", builder.reshape(env.get("_1536"), [1,sequence_length,32,2,32]));
  env.set("_1547", builder["mul"](env.get("_1537"), env.get("_1546"), {"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_mul_cos_1204"}));
  env.set("_1548", builder.reshape(env.get("_1547"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_1537"), 2, {"axis":3,"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_split_partial_input0_1196"});
    env.set("_1538", tmp[0]);
    env.set("_1539", tmp[1]);
  }
  env.set("_1540", builder.concat([env.get("_1539"), env.get("_1538")], 3, {"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1197"}));
  env.set("Inserted_615", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1188","maxValue":2047,"minValue":-2048}));
  env.set("_1527", builder["gather"](env.get("_621"), env.get("Inserted_615"), {"axis":0,"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_gather_sin_1187"}));
  env.set("_1528", builder.reshape(env.get("_1527"), [1,sequence_length,1,1,32]));
  env.set("_1541", builder["mul"](env.get("_1540"), env.get("_1528"), {"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_mul_sin_1198"}));
  env.set("_1543", builder["mul"](env.get("_1541"), env.get("_1542"), {"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_mul_sign_1199"}));
  env.set("_1544", builder.reshape(env.get("_1543"), [1,sequence_length,32,64]));
  env.set("_1549", builder["add"](env.get("_1548"), env.get("_1544"), {"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1206"}));
  env.set("_1550", builder.reshape(env.get("_1549"), [1,sequence_length,2048]));
  env.set("_1551", builder.reshape(env.get("_1550"), [1,sequence_length,32,64]));
  env.set("_1552", builder["transpose"](env.get("_1551"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/query/transpose_1209","permutation":[0,2,1,3]}));
  env.set("Inserted_576", builder.cast(env.get("_598"), "uint8"));
  env.set("_1482", builder["where"](env.get("Inserted_576"), env.get("_599"), env.get("_597"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/scatter/where_1136"}));
  env.set("_1483", builder["add"](env.get("_601"), env.get("_1482"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/right_constant/add_1138"}));
  env.set("_1484", builder.concat([env.get("_603"), env.get("_1483")], 2, {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1139"}));
  env.set("_1485", builder.reshape(env.get("_1484"), [1,sequence_length,4,3]));
  env.set("Inserted_607", builder.cast(env.get("_1485"), "int64"));
  env.set("Inserted_609", builder["max"](env.get("Inserted_607"), env.get("Inserted_608"), {"label":"Inserted_Max_1181"}));
  env.set("Inserted_611", builder["min"](env.get("Inserted_609"), env.get("Inserted_610"), {"label":"Inserted_Min_1182"}));
  env.set("Inserted_600", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1172","maxValue":2047,"minValue":-2048}));
  env.set("_1515", builder["gather"](env.get("_641"), env.get("Inserted_600"), {"axis":0,"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_gather_cos_1171"}));
  env.set("_1516", builder.reshape(env.get("_1515"), [1,sequence_length,1,1,32]));
  env.set("_1502", builder.dequantizeLinear(env.get("_1499"), env.get("_1500"), env.get("_1501"), {"axis":2,"blockSize":32,"label":"/model/layers.8/attn/k_proj/MatMul_Q4_dequantizeLinear_1160"}));
  env.set("_1503", builder.reshape(env.get("_1502"), [256,2048]));
  env.set("_1504", builder["transpose"](env.get("_1503"), {"label":"/model/layers.8/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1162","permutation":[1,0]}));
  env.set("_1505", builder["matmul"](env.get("_1479"), env.get("_1504"), {"label":"/model/layers.8/attn/k_proj/MatMul_Q4_matmul_1163"}));
  env.set("_1506", builder.reshape(env.get("_1505"), [1,sequence_length,4,64]));
  env.set("_1507", builder.reshape(env.get("_1506"), [1,sequence_length,4,2,32]));
  env.set("_1517", builder["mul"](env.get("_1507"), env.get("_1516"), {"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_mul_cos_1174"}));
  env.set("_1518", builder.reshape(env.get("_1517"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_1507"), 2, {"axis":3,"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_split_partial_input0_1166"});
    env.set("_1508", tmp[0]);
    env.set("_1509", tmp[1]);
  }
  env.set("_1510", builder.concat([env.get("_1509"), env.get("_1508")], 3, {"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1167"}));
  env.set("Inserted_591", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1158","maxValue":2047,"minValue":-2048}));
  env.set("_1497", builder["gather"](env.get("_621"), env.get("Inserted_591"), {"axis":0,"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_gather_sin_1157"}));
  env.set("_1498", builder.reshape(env.get("_1497"), [1,sequence_length,1,1,32]));
  env.set("_1511", builder["mul"](env.get("_1510"), env.get("_1498"), {"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_mul_sin_1168"}));
  env.set("_1513", builder["mul"](env.get("_1511"), env.get("_1512"), {"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_mul_sign_1169"}));
  env.set("_1514", builder.reshape(env.get("_1513"), [1,sequence_length,4,64]));
  env.set("_1519", builder["add"](env.get("_1518"), env.get("_1514"), {"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1176"}));
  env.set("_1520", builder.reshape(env.get("_1519"), [1,sequence_length,256]));
  env.set("_1521", builder.reshape(env.get("_1520"), [1,sequence_length,4,64]));
  env.set("present_8_key_16", builder["scatterND"](env.get("past_key_values_8_key_1522"), env.get("Inserted_611"), env.get("_1521"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1179"}));
  env.set("_1523", builder.reshape(env.get("present_8_key_16"), [1,4,1,past_sequence_length,64]));
  env.set("_1524", builder.expand(env.get("_1523"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/true_present_key/expand_1184"}));
  env.set("_1525", builder.reshape(env.get("_1524"), [1,32,past_sequence_length,64]));
  env.set("_1526", builder["transpose"](env.get("_1525"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/present_key/transpose_1186","permutation":[0,1,3,2]}));
  env.set("_1553", builder["matmul"](env.get("_1552"), env.get("_1526"), {"label":"/model/layers.8/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1210"}));
  env.set("_1554", builder["mul"](env.get("_1553"), env.get("_681"), {"label":"/model/layers.8/attn/GroupQueryAttention_/Attention/qkv/div_1211"}));
  env.set("_1493", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1151"}));
  env.set("_1494", builder.cumulativeSum(env.get("_1493"), 3, {"exclusive":true,"label":"/model/layers.8/attn/GroupQueryAttention_range_of_mask_shape_1152"}));
  env.set("_1490", builder["add"](env.get("_610"), env.get("_1482"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/attn_mask/add_1148"}));
  env.set("_1491", builder.expand(env.get("_1490"), [past_sequence_length,sequence_length], {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/expand_neq_right_1149"}));
  env.set("_1492", builder["transpose"](env.get("_1491"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/neq_right/transpose_1150","permutation":[1,0]}));
  env.set("Inserted_589", builder["lesser"](env.get("_1494"), env.get("_1492"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1153"}));
  env.set("_1495", builder.cast(env.get("Inserted_589"), "uint8"));
  env.set("Inserted_590", builder.cast(env.get("_1495"), "uint8"));
  env.set("_1496", builder["where"](env.get("Inserted_590"), env.get("_618"), env.get("_619"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/attn_mask/where_1155"}));
  env.set("_1555", builder["add"](env.get("_1554"), env.get("_1496"), {"label":"/model/layers.8/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1212"}));
  env.set("_1556", builder["softmax"](env.get("_1555"), 3));
  env.set("Inserted_578", builder.cast(env.get("_1485"), "int64"));
  env.set("Inserted_580", builder["max"](env.get("Inserted_578"), env.get("Inserted_579"), {"label":"Inserted_Max_1143"}));
  env.set("Inserted_582", builder["min"](env.get("Inserted_580"), env.get("Inserted_581"), {"label":"Inserted_Min_1144"}));
  env.set("_384", builder.dequantizeLinear(env.get("_381"), env.get("_382"), env.get("_383"), {"axis":2,"blockSize":32,"label":"/model/layers.8/attn/v_proj/MatMul_Q4_dequantizeLinear_168"}));
  env.set("_385", builder.reshape(env.get("_384"), [256,2048]));
  env.set("_386", builder["transpose"](env.get("_385"), {"label":"/model/layers.8/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_170","permutation":[1,0]}));
  env.set("_1480", builder["matmul"](env.get("_1479"), env.get("_386"), {"label":"/model/layers.8/attn/v_proj/MatMul_Q4_matmul_1134"}));
  env.set("_1481", builder.reshape(env.get("_1480"), [1,sequence_length,4,64]));
  env.set("present_8_value_17", builder["scatterND"](env.get("past_key_values_8_value_1486"), env.get("Inserted_582"), env.get("_1481"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1141"}));
  env.set("_1487", builder.reshape(env.get("present_8_value_17"), [1,4,1,past_sequence_length,64]));
  env.set("_1488", builder.expand(env.get("_1487"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/true_present_value/expand_1146"}));
  env.set("_1489", builder.reshape(env.get("_1488"), [1,32,past_sequence_length,64]));
  env.set("_1557", builder["matmul"](env.get("_1556"), env.get("_1489"), {"label":"/model/layers.8/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1214"}));
  env.set("_1558", builder["transpose"](env.get("_1557"), {"label":"/model/layers.8/attn/GroupQueryAttention_/Attention/qkv/transpose_1215","permutation":[0,2,1,3]}));
  env.set("_1559", builder.reshape(env.get("_1558"), [1,sequence_length,2048]));
  env.set("_378", builder.dequantizeLinear(env.get("_375"), env.get("_376"), env.get("_377"), {"axis":2,"blockSize":32,"label":"/model/layers.8/attn/o_proj/MatMul_Q4_dequantizeLinear_165"}));
  env.set("_379", builder.reshape(env.get("_378"), [2048,2048]));
  env.set("_380", builder["transpose"](env.get("_379"), {"label":"/model/layers.8/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_167","permutation":[1,0]}));
  env.set("_1560", builder["matmul"](env.get("_1559"), env.get("_380"), {"label":"/model/layers.8/attn/o_proj/MatMul_Q4_matmul_1217"}));
  env.set("_1561", builder["add"](env.get("_1472"), env.get("_1560"), {"label":"/model/layers.8/post_attention_layernorm/SkipLayerNorm_add_skip_1218"}));
  env.set("_1562", builder["pow"](env.get("_1561"), env.get("_582"), {"label":"/model/layers.8/post_attention_layernorm/SkipLayerNorm_pow_1219"}));
  env.set("_1563", builder["reduceMean"](env.get("_1562"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.8/post_attention_layernorm/SkipLayerNorm_reduceMean_1220"}));
  env.set("_1564", builder["add"](env.get("_1563"), env.get("_585"), {"label":"/model/layers.8/post_attention_layernorm/SkipLayerNorm_add_1221"}));
  env.set("_1565", builder["sqrt"](env.get("_1564"), {"label":"/model/layers.8/post_attention_layernorm/SkipLayerNorm_sqrt_1222"}));
  env.set("_1566", builder["div"](env.get("_1561"), env.get("_1565"), {"label":"/model/layers.8/post_attention_layernorm/SkipLayerNorm_div_1223"}));
  env.set("_1568", builder["mul"](env.get("_1567"), env.get("_1566"), {"label":"/model/layers.8/post_attention_layernorm/SkipLayerNorm_mul_1224"}));
  env.set("_1576", builder["matmul"](env.get("_1568"), env.get("_1575"), {"label":"/model/layers.8/mlp/gate_proj/MatMul_Q4_matmul_1229"}));
  env.set("_1577", builder["sigmoid"](env.get("_1576"), {"label":"/model/layers.8/mlp/act_fn/Sigmoid_1230"}));
  env.set("_1578", builder["mul"](env.get("_1576"), env.get("_1577"), {"label":"/model/layers.8/mlp/act_fn/Mul_1231"}));
  env.set("_372", builder.dequantizeLinear(env.get("_369"), env.get("_370"), env.get("_371"), {"axis":2,"blockSize":32,"label":"/model/layers.8/mlp/up_proj/MatMul_Q4_dequantizeLinear_162"}));
  env.set("_373", builder.reshape(env.get("_372"), [5632,2048]));
  env.set("_374", builder["transpose"](env.get("_373"), {"label":"/model/layers.8/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_164","permutation":[1,0]}));
  env.set("_1569", builder["matmul"](env.get("_1568"), env.get("_374"), {"label":"/model/layers.8/mlp/up_proj/MatMul_Q4_matmul_1225"}));
  env.set("_1579", builder["mul"](env.get("_1578"), env.get("_1569"), {"label":"/model/layers.8/mlp/Mul_1232"}));
  env.set("_366", builder.dequantizeLinear(env.get("_363"), env.get("_364"), env.get("_365"), {"axis":2,"blockSize":32,"label":"/model/layers.8/mlp/down_proj/MatMul_Q4_dequantizeLinear_159"}));
  env.set("_367", builder.reshape(env.get("_366"), [2048,5632]));
  env.set("_368", builder["transpose"](env.get("_367"), {"label":"/model/layers.8/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_161","permutation":[1,0]}));
  env.set("_1580", builder["matmul"](env.get("_1579"), env.get("_368"), {"label":"/model/layers.8/mlp/down_proj/MatMul_Q4_matmul_1233"}));
  env.set("_1581", builder["add"](env.get("_1561"), env.get("_1580"), {"label":"/model/layers.9/input_layernorm/SkipLayerNorm_add_skip_1234"}));
  env.set("_1582", builder["pow"](env.get("_1581"), env.get("_582"), {"label":"/model/layers.9/input_layernorm/SkipLayerNorm_pow_1235"}));
  env.set("_1583", builder["reduceMean"](env.get("_1582"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.9/input_layernorm/SkipLayerNorm_reduceMean_1236"}));
  env.set("_1584", builder["add"](env.get("_1583"), env.get("_585"), {"label":"/model/layers.9/input_layernorm/SkipLayerNorm_add_1237"}));
  env.set("_1585", builder["sqrt"](env.get("_1584"), {"label":"/model/layers.9/input_layernorm/SkipLayerNorm_sqrt_1238"}));
  env.set("_1586", builder["div"](env.get("_1581"), env.get("_1585"), {"label":"/model/layers.9/input_layernorm/SkipLayerNorm_div_1239"}));
  env.set("_1588", builder["mul"](env.get("_1587"), env.get("_1586"), {"label":"/model/layers.9/input_layernorm/SkipLayerNorm_mul_1240"}));
  env.set("_1644", builder["matmul"](env.get("_1588"), env.get("_1643"), {"label":"/model/layers.9/attn/q_proj/MatMul_Q4_matmul_1300"}));
  env.set("_1645", builder.reshape(env.get("_1644"), [1,sequence_length,32,64]));
  env.set("_1646", builder.reshape(env.get("_1645"), [1,sequence_length,32,2,32]));
  env.set("_1656", builder["mul"](env.get("_1646"), env.get("_1655"), {"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_mul_cos_1311"}));
  env.set("_1657", builder.reshape(env.get("_1656"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_1646"), 2, {"axis":3,"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_split_partial_input0_1303"});
    env.set("_1647", tmp[0]);
    env.set("_1648", tmp[1]);
  }
  env.set("_1649", builder.concat([env.get("_1648"), env.get("_1647")], 3, {"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1304"}));
  env.set("Inserted_675", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1295","maxValue":2047,"minValue":-2048}));
  env.set("_1636", builder["gather"](env.get("_621"), env.get("Inserted_675"), {"axis":0,"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_gather_sin_1294"}));
  env.set("_1637", builder.reshape(env.get("_1636"), [1,sequence_length,1,1,32]));
  env.set("_1650", builder["mul"](env.get("_1649"), env.get("_1637"), {"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_mul_sin_1305"}));
  env.set("_1652", builder["mul"](env.get("_1650"), env.get("_1651"), {"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_mul_sign_1306"}));
  env.set("_1653", builder.reshape(env.get("_1652"), [1,sequence_length,32,64]));
  env.set("_1658", builder["add"](env.get("_1657"), env.get("_1653"), {"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1313"}));
  env.set("_1659", builder.reshape(env.get("_1658"), [1,sequence_length,2048]));
  env.set("_1660", builder.reshape(env.get("_1659"), [1,sequence_length,32,64]));
  env.set("_1661", builder["transpose"](env.get("_1660"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/query/transpose_1316","permutation":[0,2,1,3]}));
  env.set("Inserted_636", builder.cast(env.get("_598"), "uint8"));
  env.set("_1591", builder["where"](env.get("Inserted_636"), env.get("_599"), env.get("_597"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/scatter/where_1243"}));
  env.set("_1592", builder["add"](env.get("_601"), env.get("_1591"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/right_constant/add_1245"}));
  env.set("_1593", builder.concat([env.get("_603"), env.get("_1592")], 2, {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1246"}));
  env.set("_1594", builder.reshape(env.get("_1593"), [1,sequence_length,4,3]));
  env.set("Inserted_667", builder.cast(env.get("_1594"), "int64"));
  env.set("Inserted_669", builder["max"](env.get("Inserted_667"), env.get("Inserted_668"), {"label":"Inserted_Max_1288"}));
  env.set("Inserted_671", builder["min"](env.get("Inserted_669"), env.get("Inserted_670"), {"label":"Inserted_Min_1289"}));
  env.set("Inserted_660", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1279","maxValue":2047,"minValue":-2048}));
  env.set("_1624", builder["gather"](env.get("_641"), env.get("Inserted_660"), {"axis":0,"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_gather_cos_1278"}));
  env.set("_1625", builder.reshape(env.get("_1624"), [1,sequence_length,1,1,32]));
  env.set("_1611", builder.dequantizeLinear(env.get("_1608"), env.get("_1609"), env.get("_1610"), {"axis":2,"blockSize":32,"label":"/model/layers.9/attn/k_proj/MatMul_Q4_dequantizeLinear_1267"}));
  env.set("_1612", builder.reshape(env.get("_1611"), [256,2048]));
  env.set("_1613", builder["transpose"](env.get("_1612"), {"label":"/model/layers.9/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1269","permutation":[1,0]}));
  env.set("_1614", builder["matmul"](env.get("_1588"), env.get("_1613"), {"label":"/model/layers.9/attn/k_proj/MatMul_Q4_matmul_1270"}));
  env.set("_1615", builder.reshape(env.get("_1614"), [1,sequence_length,4,64]));
  env.set("_1616", builder.reshape(env.get("_1615"), [1,sequence_length,4,2,32]));
  env.set("_1626", builder["mul"](env.get("_1616"), env.get("_1625"), {"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_mul_cos_1281"}));
  env.set("_1627", builder.reshape(env.get("_1626"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_1616"), 2, {"axis":3,"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_split_partial_input0_1273"});
    env.set("_1617", tmp[0]);
    env.set("_1618", tmp[1]);
  }
  env.set("_1619", builder.concat([env.get("_1618"), env.get("_1617")], 3, {"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1274"}));
  env.set("Inserted_651", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1265","maxValue":2047,"minValue":-2048}));
  env.set("_1606", builder["gather"](env.get("_621"), env.get("Inserted_651"), {"axis":0,"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_gather_sin_1264"}));
  env.set("_1607", builder.reshape(env.get("_1606"), [1,sequence_length,1,1,32]));
  env.set("_1620", builder["mul"](env.get("_1619"), env.get("_1607"), {"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_mul_sin_1275"}));
  env.set("_1622", builder["mul"](env.get("_1620"), env.get("_1621"), {"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_mul_sign_1276"}));
  env.set("_1623", builder.reshape(env.get("_1622"), [1,sequence_length,4,64]));
  env.set("_1628", builder["add"](env.get("_1627"), env.get("_1623"), {"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1283"}));
  env.set("_1629", builder.reshape(env.get("_1628"), [1,sequence_length,256]));
  env.set("_1630", builder.reshape(env.get("_1629"), [1,sequence_length,4,64]));
  env.set("present_9_key_18", builder["scatterND"](env.get("past_key_values_9_key_1631"), env.get("Inserted_671"), env.get("_1630"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1286"}));
  env.set("_1632", builder.reshape(env.get("present_9_key_18"), [1,4,1,past_sequence_length,64]));
  env.set("_1633", builder.expand(env.get("_1632"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/true_present_key/expand_1291"}));
  env.set("_1634", builder.reshape(env.get("_1633"), [1,32,past_sequence_length,64]));
  env.set("_1635", builder["transpose"](env.get("_1634"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/present_key/transpose_1293","permutation":[0,1,3,2]}));
  env.set("_1662", builder["matmul"](env.get("_1661"), env.get("_1635"), {"label":"/model/layers.9/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1317"}));
  env.set("_1663", builder["mul"](env.get("_1662"), env.get("_681"), {"label":"/model/layers.9/attn/GroupQueryAttention_/Attention/qkv/div_1318"}));
  env.set("_1602", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1258"}));
  env.set("_1603", builder.cumulativeSum(env.get("_1602"), 3, {"exclusive":true,"label":"/model/layers.9/attn/GroupQueryAttention_range_of_mask_shape_1259"}));
  env.set("_1599", builder["add"](env.get("_610"), env.get("_1591"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/attn_mask/add_1255"}));
  env.set("_1600", builder.expand(env.get("_1599"), [past_sequence_length,sequence_length], {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/expand_neq_right_1256"}));
  env.set("_1601", builder["transpose"](env.get("_1600"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/neq_right/transpose_1257","permutation":[1,0]}));
  env.set("Inserted_649", builder["lesser"](env.get("_1603"), env.get("_1601"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1260"}));
  env.set("_1604", builder.cast(env.get("Inserted_649"), "uint8"));
  env.set("Inserted_650", builder.cast(env.get("_1604"), "uint8"));
  env.set("_1605", builder["where"](env.get("Inserted_650"), env.get("_618"), env.get("_619"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/attn_mask/where_1262"}));
  env.set("_1664", builder["add"](env.get("_1663"), env.get("_1605"), {"label":"/model/layers.9/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1319"}));
  env.set("_1665", builder["softmax"](env.get("_1664"), 3));
  env.set("Inserted_638", builder.cast(env.get("_1594"), "int64"));
  env.set("Inserted_640", builder["max"](env.get("Inserted_638"), env.get("Inserted_639"), {"label":"Inserted_Max_1250"}));
  env.set("Inserted_642", builder["min"](env.get("Inserted_640"), env.get("Inserted_641"), {"label":"Inserted_Min_1251"}));
  env.set("_360", builder.dequantizeLinear(env.get("_357"), env.get("_358"), env.get("_359"), {"axis":2,"blockSize":32,"label":"/model/layers.9/attn/v_proj/MatMul_Q4_dequantizeLinear_156"}));
  env.set("_361", builder.reshape(env.get("_360"), [256,2048]));
  env.set("_362", builder["transpose"](env.get("_361"), {"label":"/model/layers.9/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_158","permutation":[1,0]}));
  env.set("_1589", builder["matmul"](env.get("_1588"), env.get("_362"), {"label":"/model/layers.9/attn/v_proj/MatMul_Q4_matmul_1241"}));
  env.set("_1590", builder.reshape(env.get("_1589"), [1,sequence_length,4,64]));
  env.set("present_9_value_19", builder["scatterND"](env.get("past_key_values_9_value_1595"), env.get("Inserted_642"), env.get("_1590"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1248"}));
  env.set("_1596", builder.reshape(env.get("present_9_value_19"), [1,4,1,past_sequence_length,64]));
  env.set("_1597", builder.expand(env.get("_1596"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/true_present_value/expand_1253"}));
  env.set("_1598", builder.reshape(env.get("_1597"), [1,32,past_sequence_length,64]));
  env.set("_1666", builder["matmul"](env.get("_1665"), env.get("_1598"), {"label":"/model/layers.9/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1321"}));
  env.set("_1667", builder["transpose"](env.get("_1666"), {"label":"/model/layers.9/attn/GroupQueryAttention_/Attention/qkv/transpose_1322","permutation":[0,2,1,3]}));
  env.set("_1668", builder.reshape(env.get("_1667"), [1,sequence_length,2048]));
  env.set("_354", builder.dequantizeLinear(env.get("_351"), env.get("_352"), env.get("_353"), {"axis":2,"blockSize":32,"label":"/model/layers.9/attn/o_proj/MatMul_Q4_dequantizeLinear_153"}));
  env.set("_355", builder.reshape(env.get("_354"), [2048,2048]));
  env.set("_356", builder["transpose"](env.get("_355"), {"label":"/model/layers.9/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_155","permutation":[1,0]}));
  env.set("_1669", builder["matmul"](env.get("_1668"), env.get("_356"), {"label":"/model/layers.9/attn/o_proj/MatMul_Q4_matmul_1324"}));
  env.set("_1670", builder["add"](env.get("_1581"), env.get("_1669"), {"label":"/model/layers.9/post_attention_layernorm/SkipLayerNorm_add_skip_1325"}));
  env.set("_1671", builder["pow"](env.get("_1670"), env.get("_582"), {"label":"/model/layers.9/post_attention_layernorm/SkipLayerNorm_pow_1326"}));
  env.set("_1672", builder["reduceMean"](env.get("_1671"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.9/post_attention_layernorm/SkipLayerNorm_reduceMean_1327"}));
  env.set("_1673", builder["add"](env.get("_1672"), env.get("_585"), {"label":"/model/layers.9/post_attention_layernorm/SkipLayerNorm_add_1328"}));
  env.set("_1674", builder["sqrt"](env.get("_1673"), {"label":"/model/layers.9/post_attention_layernorm/SkipLayerNorm_sqrt_1329"}));
  env.set("_1675", builder["div"](env.get("_1670"), env.get("_1674"), {"label":"/model/layers.9/post_attention_layernorm/SkipLayerNorm_div_1330"}));
  env.set("_1677", builder["mul"](env.get("_1676"), env.get("_1675"), {"label":"/model/layers.9/post_attention_layernorm/SkipLayerNorm_mul_1331"}));
  env.set("_1685", builder["matmul"](env.get("_1677"), env.get("_1684"), {"label":"/model/layers.9/mlp/gate_proj/MatMul_Q4_matmul_1336"}));
  env.set("_1686", builder["sigmoid"](env.get("_1685"), {"label":"/model/layers.9/mlp/act_fn/Sigmoid_1337"}));
  env.set("_1687", builder["mul"](env.get("_1685"), env.get("_1686"), {"label":"/model/layers.9/mlp/act_fn/Mul_1338"}));
  env.set("_348", builder.dequantizeLinear(env.get("_345"), env.get("_346"), env.get("_347"), {"axis":2,"blockSize":32,"label":"/model/layers.9/mlp/up_proj/MatMul_Q4_dequantizeLinear_150"}));
  env.set("_349", builder.reshape(env.get("_348"), [5632,2048]));
  env.set("_350", builder["transpose"](env.get("_349"), {"label":"/model/layers.9/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_152","permutation":[1,0]}));
  env.set("_1678", builder["matmul"](env.get("_1677"), env.get("_350"), {"label":"/model/layers.9/mlp/up_proj/MatMul_Q4_matmul_1332"}));
  env.set("_1688", builder["mul"](env.get("_1687"), env.get("_1678"), {"label":"/model/layers.9/mlp/Mul_1339"}));
  env.set("_342", builder.dequantizeLinear(env.get("_339"), env.get("_340"), env.get("_341"), {"axis":2,"blockSize":32,"label":"/model/layers.9/mlp/down_proj/MatMul_Q4_dequantizeLinear_147"}));
  env.set("_343", builder.reshape(env.get("_342"), [2048,5632]));
  env.set("_344", builder["transpose"](env.get("_343"), {"label":"/model/layers.9/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_149","permutation":[1,0]}));
  env.set("_1689", builder["matmul"](env.get("_1688"), env.get("_344"), {"label":"/model/layers.9/mlp/down_proj/MatMul_Q4_matmul_1340"}));
  env.set("_1690", builder["add"](env.get("_1670"), env.get("_1689"), {"label":"/model/layers.10/input_layernorm/SkipLayerNorm_add_skip_1341"}));
  env.set("_1691", builder["pow"](env.get("_1690"), env.get("_582"), {"label":"/model/layers.10/input_layernorm/SkipLayerNorm_pow_1342"}));
  env.set("_1692", builder["reduceMean"](env.get("_1691"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.10/input_layernorm/SkipLayerNorm_reduceMean_1343"}));
  env.set("_1693", builder["add"](env.get("_1692"), env.get("_585"), {"label":"/model/layers.10/input_layernorm/SkipLayerNorm_add_1344"}));
  env.set("_1694", builder["sqrt"](env.get("_1693"), {"label":"/model/layers.10/input_layernorm/SkipLayerNorm_sqrt_1345"}));
  env.set("_1695", builder["div"](env.get("_1690"), env.get("_1694"), {"label":"/model/layers.10/input_layernorm/SkipLayerNorm_div_1346"}));
  env.set("_1697", builder["mul"](env.get("_1696"), env.get("_1695"), {"label":"/model/layers.10/input_layernorm/SkipLayerNorm_mul_1347"}));
  env.set("_1753", builder["matmul"](env.get("_1697"), env.get("_1752"), {"label":"/model/layers.10/attn/q_proj/MatMul_Q4_matmul_1407"}));
  env.set("_1754", builder.reshape(env.get("_1753"), [1,sequence_length,32,64]));
  env.set("_1755", builder.reshape(env.get("_1754"), [1,sequence_length,32,2,32]));
  env.set("_1765", builder["mul"](env.get("_1755"), env.get("_1764"), {"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_mul_cos_1418"}));
  env.set("_1766", builder.reshape(env.get("_1765"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_1755"), 2, {"axis":3,"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_split_partial_input0_1410"});
    env.set("_1756", tmp[0]);
    env.set("_1757", tmp[1]);
  }
  env.set("_1758", builder.concat([env.get("_1757"), env.get("_1756")], 3, {"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1411"}));
  env.set("Inserted_735", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1402","maxValue":2047,"minValue":-2048}));
  env.set("_1745", builder["gather"](env.get("_621"), env.get("Inserted_735"), {"axis":0,"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_gather_sin_1401"}));
  env.set("_1746", builder.reshape(env.get("_1745"), [1,sequence_length,1,1,32]));
  env.set("_1759", builder["mul"](env.get("_1758"), env.get("_1746"), {"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_mul_sin_1412"}));
  env.set("_1761", builder["mul"](env.get("_1759"), env.get("_1760"), {"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_mul_sign_1413"}));
  env.set("_1762", builder.reshape(env.get("_1761"), [1,sequence_length,32,64]));
  env.set("_1767", builder["add"](env.get("_1766"), env.get("_1762"), {"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1420"}));
  env.set("_1768", builder.reshape(env.get("_1767"), [1,sequence_length,2048]));
  env.set("_1769", builder.reshape(env.get("_1768"), [1,sequence_length,32,64]));
  env.set("_1770", builder["transpose"](env.get("_1769"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/query/transpose_1423","permutation":[0,2,1,3]}));
  env.set("Inserted_696", builder.cast(env.get("_598"), "uint8"));
  env.set("_1700", builder["where"](env.get("Inserted_696"), env.get("_599"), env.get("_597"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/scatter/where_1350"}));
  env.set("_1701", builder["add"](env.get("_601"), env.get("_1700"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/right_constant/add_1352"}));
  env.set("_1702", builder.concat([env.get("_603"), env.get("_1701")], 2, {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1353"}));
  env.set("_1703", builder.reshape(env.get("_1702"), [1,sequence_length,4,3]));
  env.set("Inserted_727", builder.cast(env.get("_1703"), "int64"));
  env.set("Inserted_729", builder["max"](env.get("Inserted_727"), env.get("Inserted_728"), {"label":"Inserted_Max_1395"}));
  env.set("Inserted_731", builder["min"](env.get("Inserted_729"), env.get("Inserted_730"), {"label":"Inserted_Min_1396"}));
  env.set("Inserted_720", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1386","maxValue":2047,"minValue":-2048}));
  env.set("_1733", builder["gather"](env.get("_641"), env.get("Inserted_720"), {"axis":0,"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_gather_cos_1385"}));
  env.set("_1734", builder.reshape(env.get("_1733"), [1,sequence_length,1,1,32]));
  env.set("_1720", builder.dequantizeLinear(env.get("_1717"), env.get("_1718"), env.get("_1719"), {"axis":2,"blockSize":32,"label":"/model/layers.10/attn/k_proj/MatMul_Q4_dequantizeLinear_1374"}));
  env.set("_1721", builder.reshape(env.get("_1720"), [256,2048]));
  env.set("_1722", builder["transpose"](env.get("_1721"), {"label":"/model/layers.10/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1376","permutation":[1,0]}));
  env.set("_1723", builder["matmul"](env.get("_1697"), env.get("_1722"), {"label":"/model/layers.10/attn/k_proj/MatMul_Q4_matmul_1377"}));
  env.set("_1724", builder.reshape(env.get("_1723"), [1,sequence_length,4,64]));
  env.set("_1725", builder.reshape(env.get("_1724"), [1,sequence_length,4,2,32]));
  env.set("_1735", builder["mul"](env.get("_1725"), env.get("_1734"), {"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_mul_cos_1388"}));
  env.set("_1736", builder.reshape(env.get("_1735"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_1725"), 2, {"axis":3,"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_split_partial_input0_1380"});
    env.set("_1726", tmp[0]);
    env.set("_1727", tmp[1]);
  }
  env.set("_1728", builder.concat([env.get("_1727"), env.get("_1726")], 3, {"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1381"}));
  env.set("Inserted_711", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1372","maxValue":2047,"minValue":-2048}));
  env.set("_1715", builder["gather"](env.get("_621"), env.get("Inserted_711"), {"axis":0,"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_gather_sin_1371"}));
  env.set("_1716", builder.reshape(env.get("_1715"), [1,sequence_length,1,1,32]));
  env.set("_1729", builder["mul"](env.get("_1728"), env.get("_1716"), {"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_mul_sin_1382"}));
  env.set("_1731", builder["mul"](env.get("_1729"), env.get("_1730"), {"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_mul_sign_1383"}));
  env.set("_1732", builder.reshape(env.get("_1731"), [1,sequence_length,4,64]));
  env.set("_1737", builder["add"](env.get("_1736"), env.get("_1732"), {"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1390"}));
  env.set("_1738", builder.reshape(env.get("_1737"), [1,sequence_length,256]));
  env.set("_1739", builder.reshape(env.get("_1738"), [1,sequence_length,4,64]));
  env.set("present_10_key_20", builder["scatterND"](env.get("past_key_values_10_key_1740"), env.get("Inserted_731"), env.get("_1739"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1393"}));
  env.set("_1741", builder.reshape(env.get("present_10_key_20"), [1,4,1,past_sequence_length,64]));
  env.set("_1742", builder.expand(env.get("_1741"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/true_present_key/expand_1398"}));
  env.set("_1743", builder.reshape(env.get("_1742"), [1,32,past_sequence_length,64]));
  env.set("_1744", builder["transpose"](env.get("_1743"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/present_key/transpose_1400","permutation":[0,1,3,2]}));
  env.set("_1771", builder["matmul"](env.get("_1770"), env.get("_1744"), {"label":"/model/layers.10/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1424"}));
  env.set("_1772", builder["mul"](env.get("_1771"), env.get("_681"), {"label":"/model/layers.10/attn/GroupQueryAttention_/Attention/qkv/div_1425"}));
  env.set("_1711", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1365"}));
  env.set("_1712", builder.cumulativeSum(env.get("_1711"), 3, {"exclusive":true,"label":"/model/layers.10/attn/GroupQueryAttention_range_of_mask_shape_1366"}));
  env.set("_1708", builder["add"](env.get("_610"), env.get("_1700"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/attn_mask/add_1362"}));
  env.set("_1709", builder.expand(env.get("_1708"), [past_sequence_length,sequence_length], {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/expand_neq_right_1363"}));
  env.set("_1710", builder["transpose"](env.get("_1709"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/neq_right/transpose_1364","permutation":[1,0]}));
  env.set("Inserted_709", builder["lesser"](env.get("_1712"), env.get("_1710"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1367"}));
  env.set("_1713", builder.cast(env.get("Inserted_709"), "uint8"));
  env.set("Inserted_710", builder.cast(env.get("_1713"), "uint8"));
  env.set("_1714", builder["where"](env.get("Inserted_710"), env.get("_618"), env.get("_619"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/attn_mask/where_1369"}));
  env.set("_1773", builder["add"](env.get("_1772"), env.get("_1714"), {"label":"/model/layers.10/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1426"}));
  env.set("_1774", builder["softmax"](env.get("_1773"), 3));
  env.set("Inserted_698", builder.cast(env.get("_1703"), "int64"));
  env.set("Inserted_700", builder["max"](env.get("Inserted_698"), env.get("Inserted_699"), {"label":"Inserted_Max_1357"}));
  env.set("Inserted_702", builder["min"](env.get("Inserted_700"), env.get("Inserted_701"), {"label":"Inserted_Min_1358"}));
  env.set("_336", builder.dequantizeLinear(env.get("_333"), env.get("_334"), env.get("_335"), {"axis":2,"blockSize":32,"label":"/model/layers.10/attn/v_proj/MatMul_Q4_dequantizeLinear_144"}));
  env.set("_337", builder.reshape(env.get("_336"), [256,2048]));
  env.set("_338", builder["transpose"](env.get("_337"), {"label":"/model/layers.10/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_146","permutation":[1,0]}));
  env.set("_1698", builder["matmul"](env.get("_1697"), env.get("_338"), {"label":"/model/layers.10/attn/v_proj/MatMul_Q4_matmul_1348"}));
  env.set("_1699", builder.reshape(env.get("_1698"), [1,sequence_length,4,64]));
  env.set("present_10_value_21", builder["scatterND"](env.get("past_key_values_10_value_1704"), env.get("Inserted_702"), env.get("_1699"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1355"}));
  env.set("_1705", builder.reshape(env.get("present_10_value_21"), [1,4,1,past_sequence_length,64]));
  env.set("_1706", builder.expand(env.get("_1705"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/true_present_value/expand_1360"}));
  env.set("_1707", builder.reshape(env.get("_1706"), [1,32,past_sequence_length,64]));
  env.set("_1775", builder["matmul"](env.get("_1774"), env.get("_1707"), {"label":"/model/layers.10/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1428"}));
  env.set("_1776", builder["transpose"](env.get("_1775"), {"label":"/model/layers.10/attn/GroupQueryAttention_/Attention/qkv/transpose_1429","permutation":[0,2,1,3]}));
  env.set("_1777", builder.reshape(env.get("_1776"), [1,sequence_length,2048]));
  env.set("_330", builder.dequantizeLinear(env.get("_327"), env.get("_328"), env.get("_329"), {"axis":2,"blockSize":32,"label":"/model/layers.10/attn/o_proj/MatMul_Q4_dequantizeLinear_141"}));
  env.set("_331", builder.reshape(env.get("_330"), [2048,2048]));
  env.set("_332", builder["transpose"](env.get("_331"), {"label":"/model/layers.10/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_143","permutation":[1,0]}));
  env.set("_1778", builder["matmul"](env.get("_1777"), env.get("_332"), {"label":"/model/layers.10/attn/o_proj/MatMul_Q4_matmul_1431"}));
  env.set("_1779", builder["add"](env.get("_1690"), env.get("_1778"), {"label":"/model/layers.10/post_attention_layernorm/SkipLayerNorm_add_skip_1432"}));
  env.set("_1780", builder["pow"](env.get("_1779"), env.get("_582"), {"label":"/model/layers.10/post_attention_layernorm/SkipLayerNorm_pow_1433"}));
  env.set("_1781", builder["reduceMean"](env.get("_1780"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.10/post_attention_layernorm/SkipLayerNorm_reduceMean_1434"}));
  env.set("_1782", builder["add"](env.get("_1781"), env.get("_585"), {"label":"/model/layers.10/post_attention_layernorm/SkipLayerNorm_add_1435"}));
  env.set("_1783", builder["sqrt"](env.get("_1782"), {"label":"/model/layers.10/post_attention_layernorm/SkipLayerNorm_sqrt_1436"}));
  env.set("_1784", builder["div"](env.get("_1779"), env.get("_1783"), {"label":"/model/layers.10/post_attention_layernorm/SkipLayerNorm_div_1437"}));
  env.set("_1786", builder["mul"](env.get("_1785"), env.get("_1784"), {"label":"/model/layers.10/post_attention_layernorm/SkipLayerNorm_mul_1438"}));
  env.set("_1794", builder["matmul"](env.get("_1786"), env.get("_1793"), {"label":"/model/layers.10/mlp/gate_proj/MatMul_Q4_matmul_1443"}));
  env.set("_1795", builder["sigmoid"](env.get("_1794"), {"label":"/model/layers.10/mlp/act_fn/Sigmoid_1444"}));
  env.set("_1796", builder["mul"](env.get("_1794"), env.get("_1795"), {"label":"/model/layers.10/mlp/act_fn/Mul_1445"}));
  env.set("_324", builder.dequantizeLinear(env.get("_321"), env.get("_322"), env.get("_323"), {"axis":2,"blockSize":32,"label":"/model/layers.10/mlp/up_proj/MatMul_Q4_dequantizeLinear_138"}));
  env.set("_325", builder.reshape(env.get("_324"), [5632,2048]));
  env.set("_326", builder["transpose"](env.get("_325"), {"label":"/model/layers.10/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_140","permutation":[1,0]}));
  env.set("_1787", builder["matmul"](env.get("_1786"), env.get("_326"), {"label":"/model/layers.10/mlp/up_proj/MatMul_Q4_matmul_1439"}));
  env.set("_1797", builder["mul"](env.get("_1796"), env.get("_1787"), {"label":"/model/layers.10/mlp/Mul_1446"}));
  env.set("_318", builder.dequantizeLinear(env.get("_315"), env.get("_316"), env.get("_317"), {"axis":2,"blockSize":32,"label":"/model/layers.10/mlp/down_proj/MatMul_Q4_dequantizeLinear_135"}));
  env.set("_319", builder.reshape(env.get("_318"), [2048,5632]));
  env.set("_320", builder["transpose"](env.get("_319"), {"label":"/model/layers.10/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_137","permutation":[1,0]}));
  env.set("_1798", builder["matmul"](env.get("_1797"), env.get("_320"), {"label":"/model/layers.10/mlp/down_proj/MatMul_Q4_matmul_1447"}));
  env.set("_1799", builder["add"](env.get("_1779"), env.get("_1798"), {"label":"/model/layers.11/input_layernorm/SkipLayerNorm_add_skip_1448"}));
  env.set("_1800", builder["pow"](env.get("_1799"), env.get("_582"), {"label":"/model/layers.11/input_layernorm/SkipLayerNorm_pow_1449"}));
  env.set("_1801", builder["reduceMean"](env.get("_1800"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.11/input_layernorm/SkipLayerNorm_reduceMean_1450"}));
  env.set("_1802", builder["add"](env.get("_1801"), env.get("_585"), {"label":"/model/layers.11/input_layernorm/SkipLayerNorm_add_1451"}));
  env.set("_1803", builder["sqrt"](env.get("_1802"), {"label":"/model/layers.11/input_layernorm/SkipLayerNorm_sqrt_1452"}));
  env.set("_1804", builder["div"](env.get("_1799"), env.get("_1803"), {"label":"/model/layers.11/input_layernorm/SkipLayerNorm_div_1453"}));
  env.set("_1806", builder["mul"](env.get("_1805"), env.get("_1804"), {"label":"/model/layers.11/input_layernorm/SkipLayerNorm_mul_1454"}));
  env.set("_1862", builder["matmul"](env.get("_1806"), env.get("_1861"), {"label":"/model/layers.11/attn/q_proj/MatMul_Q4_matmul_1514"}));
  env.set("_1863", builder.reshape(env.get("_1862"), [1,sequence_length,32,64]));
  env.set("_1864", builder.reshape(env.get("_1863"), [1,sequence_length,32,2,32]));
  env.set("_1874", builder["mul"](env.get("_1864"), env.get("_1873"), {"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_mul_cos_1525"}));
  env.set("_1875", builder.reshape(env.get("_1874"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_1864"), 2, {"axis":3,"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_split_partial_input0_1517"});
    env.set("_1865", tmp[0]);
    env.set("_1866", tmp[1]);
  }
  env.set("_1867", builder.concat([env.get("_1866"), env.get("_1865")], 3, {"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1518"}));
  env.set("Inserted_795", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1509","maxValue":2047,"minValue":-2048}));
  env.set("_1854", builder["gather"](env.get("_621"), env.get("Inserted_795"), {"axis":0,"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_gather_sin_1508"}));
  env.set("_1855", builder.reshape(env.get("_1854"), [1,sequence_length,1,1,32]));
  env.set("_1868", builder["mul"](env.get("_1867"), env.get("_1855"), {"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_mul_sin_1519"}));
  env.set("_1870", builder["mul"](env.get("_1868"), env.get("_1869"), {"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_mul_sign_1520"}));
  env.set("_1871", builder.reshape(env.get("_1870"), [1,sequence_length,32,64]));
  env.set("_1876", builder["add"](env.get("_1875"), env.get("_1871"), {"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1527"}));
  env.set("_1877", builder.reshape(env.get("_1876"), [1,sequence_length,2048]));
  env.set("_1878", builder.reshape(env.get("_1877"), [1,sequence_length,32,64]));
  env.set("_1879", builder["transpose"](env.get("_1878"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/query/transpose_1530","permutation":[0,2,1,3]}));
  env.set("Inserted_756", builder.cast(env.get("_598"), "uint8"));
  env.set("_1809", builder["where"](env.get("Inserted_756"), env.get("_599"), env.get("_597"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/scatter/where_1457"}));
  env.set("_1810", builder["add"](env.get("_601"), env.get("_1809"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/right_constant/add_1459"}));
  env.set("_1811", builder.concat([env.get("_603"), env.get("_1810")], 2, {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1460"}));
  env.set("_1812", builder.reshape(env.get("_1811"), [1,sequence_length,4,3]));
  env.set("Inserted_787", builder.cast(env.get("_1812"), "int64"));
  env.set("Inserted_789", builder["max"](env.get("Inserted_787"), env.get("Inserted_788"), {"label":"Inserted_Max_1502"}));
  env.set("Inserted_791", builder["min"](env.get("Inserted_789"), env.get("Inserted_790"), {"label":"Inserted_Min_1503"}));
  env.set("Inserted_780", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1493","maxValue":2047,"minValue":-2048}));
  env.set("_1842", builder["gather"](env.get("_641"), env.get("Inserted_780"), {"axis":0,"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_gather_cos_1492"}));
  env.set("_1843", builder.reshape(env.get("_1842"), [1,sequence_length,1,1,32]));
  env.set("_1829", builder.dequantizeLinear(env.get("_1826"), env.get("_1827"), env.get("_1828"), {"axis":2,"blockSize":32,"label":"/model/layers.11/attn/k_proj/MatMul_Q4_dequantizeLinear_1481"}));
  env.set("_1830", builder.reshape(env.get("_1829"), [256,2048]));
  env.set("_1831", builder["transpose"](env.get("_1830"), {"label":"/model/layers.11/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1483","permutation":[1,0]}));
  env.set("_1832", builder["matmul"](env.get("_1806"), env.get("_1831"), {"label":"/model/layers.11/attn/k_proj/MatMul_Q4_matmul_1484"}));
  env.set("_1833", builder.reshape(env.get("_1832"), [1,sequence_length,4,64]));
  env.set("_1834", builder.reshape(env.get("_1833"), [1,sequence_length,4,2,32]));
  env.set("_1844", builder["mul"](env.get("_1834"), env.get("_1843"), {"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_mul_cos_1495"}));
  env.set("_1845", builder.reshape(env.get("_1844"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_1834"), 2, {"axis":3,"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_split_partial_input0_1487"});
    env.set("_1835", tmp[0]);
    env.set("_1836", tmp[1]);
  }
  env.set("_1837", builder.concat([env.get("_1836"), env.get("_1835")], 3, {"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1488"}));
  env.set("Inserted_771", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1479","maxValue":2047,"minValue":-2048}));
  env.set("_1824", builder["gather"](env.get("_621"), env.get("Inserted_771"), {"axis":0,"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_gather_sin_1478"}));
  env.set("_1825", builder.reshape(env.get("_1824"), [1,sequence_length,1,1,32]));
  env.set("_1838", builder["mul"](env.get("_1837"), env.get("_1825"), {"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_mul_sin_1489"}));
  env.set("_1840", builder["mul"](env.get("_1838"), env.get("_1839"), {"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_mul_sign_1490"}));
  env.set("_1841", builder.reshape(env.get("_1840"), [1,sequence_length,4,64]));
  env.set("_1846", builder["add"](env.get("_1845"), env.get("_1841"), {"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1497"}));
  env.set("_1847", builder.reshape(env.get("_1846"), [1,sequence_length,256]));
  env.set("_1848", builder.reshape(env.get("_1847"), [1,sequence_length,4,64]));
  env.set("present_11_key_22", builder["scatterND"](env.get("past_key_values_11_key_1849"), env.get("Inserted_791"), env.get("_1848"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1500"}));
  env.set("_1850", builder.reshape(env.get("present_11_key_22"), [1,4,1,past_sequence_length,64]));
  env.set("_1851", builder.expand(env.get("_1850"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/true_present_key/expand_1505"}));
  env.set("_1852", builder.reshape(env.get("_1851"), [1,32,past_sequence_length,64]));
  env.set("_1853", builder["transpose"](env.get("_1852"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/present_key/transpose_1507","permutation":[0,1,3,2]}));
  env.set("_1880", builder["matmul"](env.get("_1879"), env.get("_1853"), {"label":"/model/layers.11/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1531"}));
  env.set("_1881", builder["mul"](env.get("_1880"), env.get("_681"), {"label":"/model/layers.11/attn/GroupQueryAttention_/Attention/qkv/div_1532"}));
  env.set("_1820", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1472"}));
  env.set("_1821", builder.cumulativeSum(env.get("_1820"), 3, {"exclusive":true,"label":"/model/layers.11/attn/GroupQueryAttention_range_of_mask_shape_1473"}));
  env.set("_1817", builder["add"](env.get("_610"), env.get("_1809"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/attn_mask/add_1469"}));
  env.set("_1818", builder.expand(env.get("_1817"), [past_sequence_length,sequence_length], {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/expand_neq_right_1470"}));
  env.set("_1819", builder["transpose"](env.get("_1818"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/neq_right/transpose_1471","permutation":[1,0]}));
  env.set("Inserted_769", builder["lesser"](env.get("_1821"), env.get("_1819"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1474"}));
  env.set("_1822", builder.cast(env.get("Inserted_769"), "uint8"));
  env.set("Inserted_770", builder.cast(env.get("_1822"), "uint8"));
  env.set("_1823", builder["where"](env.get("Inserted_770"), env.get("_618"), env.get("_619"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/attn_mask/where_1476"}));
  env.set("_1882", builder["add"](env.get("_1881"), env.get("_1823"), {"label":"/model/layers.11/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1533"}));
  env.set("_1883", builder["softmax"](env.get("_1882"), 3));
  env.set("Inserted_758", builder.cast(env.get("_1812"), "int64"));
  env.set("Inserted_760", builder["max"](env.get("Inserted_758"), env.get("Inserted_759"), {"label":"Inserted_Max_1464"}));
  env.set("Inserted_762", builder["min"](env.get("Inserted_760"), env.get("Inserted_761"), {"label":"Inserted_Min_1465"}));
  env.set("_312", builder.dequantizeLinear(env.get("_309"), env.get("_310"), env.get("_311"), {"axis":2,"blockSize":32,"label":"/model/layers.11/attn/v_proj/MatMul_Q4_dequantizeLinear_132"}));
  env.set("_313", builder.reshape(env.get("_312"), [256,2048]));
  env.set("_314", builder["transpose"](env.get("_313"), {"label":"/model/layers.11/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_134","permutation":[1,0]}));
  env.set("_1807", builder["matmul"](env.get("_1806"), env.get("_314"), {"label":"/model/layers.11/attn/v_proj/MatMul_Q4_matmul_1455"}));
  env.set("_1808", builder.reshape(env.get("_1807"), [1,sequence_length,4,64]));
  env.set("present_11_value_23", builder["scatterND"](env.get("past_key_values_11_value_1813"), env.get("Inserted_762"), env.get("_1808"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1462"}));
  env.set("_1814", builder.reshape(env.get("present_11_value_23"), [1,4,1,past_sequence_length,64]));
  env.set("_1815", builder.expand(env.get("_1814"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/true_present_value/expand_1467"}));
  env.set("_1816", builder.reshape(env.get("_1815"), [1,32,past_sequence_length,64]));
  env.set("_1884", builder["matmul"](env.get("_1883"), env.get("_1816"), {"label":"/model/layers.11/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1535"}));
  env.set("_1885", builder["transpose"](env.get("_1884"), {"label":"/model/layers.11/attn/GroupQueryAttention_/Attention/qkv/transpose_1536","permutation":[0,2,1,3]}));
  env.set("_1886", builder.reshape(env.get("_1885"), [1,sequence_length,2048]));
  env.set("_306", builder.dequantizeLinear(env.get("_303"), env.get("_304"), env.get("_305"), {"axis":2,"blockSize":32,"label":"/model/layers.11/attn/o_proj/MatMul_Q4_dequantizeLinear_129"}));
  env.set("_307", builder.reshape(env.get("_306"), [2048,2048]));
  env.set("_308", builder["transpose"](env.get("_307"), {"label":"/model/layers.11/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_131","permutation":[1,0]}));
  env.set("_1887", builder["matmul"](env.get("_1886"), env.get("_308"), {"label":"/model/layers.11/attn/o_proj/MatMul_Q4_matmul_1538"}));
  env.set("_1888", builder["add"](env.get("_1799"), env.get("_1887"), {"label":"/model/layers.11/post_attention_layernorm/SkipLayerNorm_add_skip_1539"}));
  env.set("_1889", builder["pow"](env.get("_1888"), env.get("_582"), {"label":"/model/layers.11/post_attention_layernorm/SkipLayerNorm_pow_1540"}));
  env.set("_1890", builder["reduceMean"](env.get("_1889"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.11/post_attention_layernorm/SkipLayerNorm_reduceMean_1541"}));
  env.set("_1891", builder["add"](env.get("_1890"), env.get("_585"), {"label":"/model/layers.11/post_attention_layernorm/SkipLayerNorm_add_1542"}));
  env.set("_1892", builder["sqrt"](env.get("_1891"), {"label":"/model/layers.11/post_attention_layernorm/SkipLayerNorm_sqrt_1543"}));
  env.set("_1893", builder["div"](env.get("_1888"), env.get("_1892"), {"label":"/model/layers.11/post_attention_layernorm/SkipLayerNorm_div_1544"}));
  env.set("_1895", builder["mul"](env.get("_1894"), env.get("_1893"), {"label":"/model/layers.11/post_attention_layernorm/SkipLayerNorm_mul_1545"}));
  env.set("_1903", builder["matmul"](env.get("_1895"), env.get("_1902"), {"label":"/model/layers.11/mlp/gate_proj/MatMul_Q4_matmul_1550"}));
  env.set("_1904", builder["sigmoid"](env.get("_1903"), {"label":"/model/layers.11/mlp/act_fn/Sigmoid_1551"}));
  env.set("_1905", builder["mul"](env.get("_1903"), env.get("_1904"), {"label":"/model/layers.11/mlp/act_fn/Mul_1552"}));
  env.set("_300", builder.dequantizeLinear(env.get("_297"), env.get("_298"), env.get("_299"), {"axis":2,"blockSize":32,"label":"/model/layers.11/mlp/up_proj/MatMul_Q4_dequantizeLinear_126"}));
  env.set("_301", builder.reshape(env.get("_300"), [5632,2048]));
  env.set("_302", builder["transpose"](env.get("_301"), {"label":"/model/layers.11/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_128","permutation":[1,0]}));
  env.set("_1896", builder["matmul"](env.get("_1895"), env.get("_302"), {"label":"/model/layers.11/mlp/up_proj/MatMul_Q4_matmul_1546"}));
  env.set("_1906", builder["mul"](env.get("_1905"), env.get("_1896"), {"label":"/model/layers.11/mlp/Mul_1553"}));
  env.set("_294", builder.dequantizeLinear(env.get("_291"), env.get("_292"), env.get("_293"), {"axis":2,"blockSize":32,"label":"/model/layers.11/mlp/down_proj/MatMul_Q4_dequantizeLinear_123"}));
  env.set("_295", builder.reshape(env.get("_294"), [2048,5632]));
  env.set("_296", builder["transpose"](env.get("_295"), {"label":"/model/layers.11/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_125","permutation":[1,0]}));
  env.set("_1907", builder["matmul"](env.get("_1906"), env.get("_296"), {"label":"/model/layers.11/mlp/down_proj/MatMul_Q4_matmul_1554"}));
  env.set("_1908", builder["add"](env.get("_1888"), env.get("_1907"), {"label":"/model/layers.12/input_layernorm/SkipLayerNorm_add_skip_1555"}));
  env.set("_1909", builder["pow"](env.get("_1908"), env.get("_582"), {"label":"/model/layers.12/input_layernorm/SkipLayerNorm_pow_1556"}));
  env.set("_1910", builder["reduceMean"](env.get("_1909"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.12/input_layernorm/SkipLayerNorm_reduceMean_1557"}));
  env.set("_1911", builder["add"](env.get("_1910"), env.get("_585"), {"label":"/model/layers.12/input_layernorm/SkipLayerNorm_add_1558"}));
  env.set("_1912", builder["sqrt"](env.get("_1911"), {"label":"/model/layers.12/input_layernorm/SkipLayerNorm_sqrt_1559"}));
  env.set("_1913", builder["div"](env.get("_1908"), env.get("_1912"), {"label":"/model/layers.12/input_layernorm/SkipLayerNorm_div_1560"}));
  env.set("_1915", builder["mul"](env.get("_1914"), env.get("_1913"), {"label":"/model/layers.12/input_layernorm/SkipLayerNorm_mul_1561"}));
  env.set("_1971", builder["matmul"](env.get("_1915"), env.get("_1970"), {"label":"/model/layers.12/attn/q_proj/MatMul_Q4_matmul_1621"}));
  env.set("_1972", builder.reshape(env.get("_1971"), [1,sequence_length,32,64]));
  env.set("_1973", builder.reshape(env.get("_1972"), [1,sequence_length,32,2,32]));
  env.set("_1983", builder["mul"](env.get("_1973"), env.get("_1982"), {"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_mul_cos_1632"}));
  env.set("_1984", builder.reshape(env.get("_1983"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_1973"), 2, {"axis":3,"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_split_partial_input0_1624"});
    env.set("_1974", tmp[0]);
    env.set("_1975", tmp[1]);
  }
  env.set("_1976", builder.concat([env.get("_1975"), env.get("_1974")], 3, {"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1625"}));
  env.set("Inserted_855", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1616","maxValue":2047,"minValue":-2048}));
  env.set("_1963", builder["gather"](env.get("_621"), env.get("Inserted_855"), {"axis":0,"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_gather_sin_1615"}));
  env.set("_1964", builder.reshape(env.get("_1963"), [1,sequence_length,1,1,32]));
  env.set("_1977", builder["mul"](env.get("_1976"), env.get("_1964"), {"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_mul_sin_1626"}));
  env.set("_1979", builder["mul"](env.get("_1977"), env.get("_1978"), {"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_mul_sign_1627"}));
  env.set("_1980", builder.reshape(env.get("_1979"), [1,sequence_length,32,64]));
  env.set("_1985", builder["add"](env.get("_1984"), env.get("_1980"), {"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1634"}));
  env.set("_1986", builder.reshape(env.get("_1985"), [1,sequence_length,2048]));
  env.set("_1987", builder.reshape(env.get("_1986"), [1,sequence_length,32,64]));
  env.set("_1988", builder["transpose"](env.get("_1987"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/query/transpose_1637","permutation":[0,2,1,3]}));
  env.set("Inserted_816", builder.cast(env.get("_598"), "uint8"));
  env.set("_1918", builder["where"](env.get("Inserted_816"), env.get("_599"), env.get("_597"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/scatter/where_1564"}));
  env.set("_1919", builder["add"](env.get("_601"), env.get("_1918"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/right_constant/add_1566"}));
  env.set("_1920", builder.concat([env.get("_603"), env.get("_1919")], 2, {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1567"}));
  env.set("_1921", builder.reshape(env.get("_1920"), [1,sequence_length,4,3]));
  env.set("Inserted_847", builder.cast(env.get("_1921"), "int64"));
  env.set("Inserted_849", builder["max"](env.get("Inserted_847"), env.get("Inserted_848"), {"label":"Inserted_Max_1609"}));
  env.set("Inserted_851", builder["min"](env.get("Inserted_849"), env.get("Inserted_850"), {"label":"Inserted_Min_1610"}));
  env.set("Inserted_840", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1600","maxValue":2047,"minValue":-2048}));
  env.set("_1951", builder["gather"](env.get("_641"), env.get("Inserted_840"), {"axis":0,"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_gather_cos_1599"}));
  env.set("_1952", builder.reshape(env.get("_1951"), [1,sequence_length,1,1,32]));
  env.set("_1938", builder.dequantizeLinear(env.get("_1935"), env.get("_1936"), env.get("_1937"), {"axis":2,"blockSize":32,"label":"/model/layers.12/attn/k_proj/MatMul_Q4_dequantizeLinear_1588"}));
  env.set("_1939", builder.reshape(env.get("_1938"), [256,2048]));
  env.set("_1940", builder["transpose"](env.get("_1939"), {"label":"/model/layers.12/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1590","permutation":[1,0]}));
  env.set("_1941", builder["matmul"](env.get("_1915"), env.get("_1940"), {"label":"/model/layers.12/attn/k_proj/MatMul_Q4_matmul_1591"}));
  env.set("_1942", builder.reshape(env.get("_1941"), [1,sequence_length,4,64]));
  env.set("_1943", builder.reshape(env.get("_1942"), [1,sequence_length,4,2,32]));
  env.set("_1953", builder["mul"](env.get("_1943"), env.get("_1952"), {"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_mul_cos_1602"}));
  env.set("_1954", builder.reshape(env.get("_1953"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_1943"), 2, {"axis":3,"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_split_partial_input0_1594"});
    env.set("_1944", tmp[0]);
    env.set("_1945", tmp[1]);
  }
  env.set("_1946", builder.concat([env.get("_1945"), env.get("_1944")], 3, {"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1595"}));
  env.set("Inserted_831", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1586","maxValue":2047,"minValue":-2048}));
  env.set("_1933", builder["gather"](env.get("_621"), env.get("Inserted_831"), {"axis":0,"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_gather_sin_1585"}));
  env.set("_1934", builder.reshape(env.get("_1933"), [1,sequence_length,1,1,32]));
  env.set("_1947", builder["mul"](env.get("_1946"), env.get("_1934"), {"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_mul_sin_1596"}));
  env.set("_1949", builder["mul"](env.get("_1947"), env.get("_1948"), {"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_mul_sign_1597"}));
  env.set("_1950", builder.reshape(env.get("_1949"), [1,sequence_length,4,64]));
  env.set("_1955", builder["add"](env.get("_1954"), env.get("_1950"), {"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1604"}));
  env.set("_1956", builder.reshape(env.get("_1955"), [1,sequence_length,256]));
  env.set("_1957", builder.reshape(env.get("_1956"), [1,sequence_length,4,64]));
  env.set("present_12_key_24", builder["scatterND"](env.get("past_key_values_12_key_1958"), env.get("Inserted_851"), env.get("_1957"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1607"}));
  env.set("_1959", builder.reshape(env.get("present_12_key_24"), [1,4,1,past_sequence_length,64]));
  env.set("_1960", builder.expand(env.get("_1959"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/true_present_key/expand_1612"}));
  env.set("_1961", builder.reshape(env.get("_1960"), [1,32,past_sequence_length,64]));
  env.set("_1962", builder["transpose"](env.get("_1961"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/present_key/transpose_1614","permutation":[0,1,3,2]}));
  env.set("_1989", builder["matmul"](env.get("_1988"), env.get("_1962"), {"label":"/model/layers.12/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1638"}));
  env.set("_1990", builder["mul"](env.get("_1989"), env.get("_681"), {"label":"/model/layers.12/attn/GroupQueryAttention_/Attention/qkv/div_1639"}));
  env.set("_1929", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1579"}));
  env.set("_1930", builder.cumulativeSum(env.get("_1929"), 3, {"exclusive":true,"label":"/model/layers.12/attn/GroupQueryAttention_range_of_mask_shape_1580"}));
  env.set("_1926", builder["add"](env.get("_610"), env.get("_1918"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/attn_mask/add_1576"}));
  env.set("_1927", builder.expand(env.get("_1926"), [past_sequence_length,sequence_length], {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/expand_neq_right_1577"}));
  env.set("_1928", builder["transpose"](env.get("_1927"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/neq_right/transpose_1578","permutation":[1,0]}));
  env.set("Inserted_829", builder["lesser"](env.get("_1930"), env.get("_1928"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1581"}));
  env.set("_1931", builder.cast(env.get("Inserted_829"), "uint8"));
  env.set("Inserted_830", builder.cast(env.get("_1931"), "uint8"));
  env.set("_1932", builder["where"](env.get("Inserted_830"), env.get("_618"), env.get("_619"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/attn_mask/where_1583"}));
  env.set("_1991", builder["add"](env.get("_1990"), env.get("_1932"), {"label":"/model/layers.12/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1640"}));
  env.set("_1992", builder["softmax"](env.get("_1991"), 3));
  env.set("Inserted_818", builder.cast(env.get("_1921"), "int64"));
  env.set("Inserted_820", builder["max"](env.get("Inserted_818"), env.get("Inserted_819"), {"label":"Inserted_Max_1571"}));
  env.set("Inserted_822", builder["min"](env.get("Inserted_820"), env.get("Inserted_821"), {"label":"Inserted_Min_1572"}));
  env.set("_288", builder.dequantizeLinear(env.get("_285"), env.get("_286"), env.get("_287"), {"axis":2,"blockSize":32,"label":"/model/layers.12/attn/v_proj/MatMul_Q4_dequantizeLinear_120"}));
  env.set("_289", builder.reshape(env.get("_288"), [256,2048]));
  env.set("_290", builder["transpose"](env.get("_289"), {"label":"/model/layers.12/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_122","permutation":[1,0]}));
  env.set("_1916", builder["matmul"](env.get("_1915"), env.get("_290"), {"label":"/model/layers.12/attn/v_proj/MatMul_Q4_matmul_1562"}));
  env.set("_1917", builder.reshape(env.get("_1916"), [1,sequence_length,4,64]));
  env.set("present_12_value_25", builder["scatterND"](env.get("past_key_values_12_value_1922"), env.get("Inserted_822"), env.get("_1917"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1569"}));
  env.set("_1923", builder.reshape(env.get("present_12_value_25"), [1,4,1,past_sequence_length,64]));
  env.set("_1924", builder.expand(env.get("_1923"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/true_present_value/expand_1574"}));
  env.set("_1925", builder.reshape(env.get("_1924"), [1,32,past_sequence_length,64]));
  env.set("_1993", builder["matmul"](env.get("_1992"), env.get("_1925"), {"label":"/model/layers.12/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1642"}));
  env.set("_1994", builder["transpose"](env.get("_1993"), {"label":"/model/layers.12/attn/GroupQueryAttention_/Attention/qkv/transpose_1643","permutation":[0,2,1,3]}));
  env.set("_1995", builder.reshape(env.get("_1994"), [1,sequence_length,2048]));
  env.set("_282", builder.dequantizeLinear(env.get("_279"), env.get("_280"), env.get("_281"), {"axis":2,"blockSize":32,"label":"/model/layers.12/attn/o_proj/MatMul_Q4_dequantizeLinear_117"}));
  env.set("_283", builder.reshape(env.get("_282"), [2048,2048]));
  env.set("_284", builder["transpose"](env.get("_283"), {"label":"/model/layers.12/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_119","permutation":[1,0]}));
  env.set("_1996", builder["matmul"](env.get("_1995"), env.get("_284"), {"label":"/model/layers.12/attn/o_proj/MatMul_Q4_matmul_1645"}));
  env.set("_1997", builder["add"](env.get("_1908"), env.get("_1996"), {"label":"/model/layers.12/post_attention_layernorm/SkipLayerNorm_add_skip_1646"}));
  env.set("_1998", builder["pow"](env.get("_1997"), env.get("_582"), {"label":"/model/layers.12/post_attention_layernorm/SkipLayerNorm_pow_1647"}));
  env.set("_1999", builder["reduceMean"](env.get("_1998"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.12/post_attention_layernorm/SkipLayerNorm_reduceMean_1648"}));
  env.set("_2000", builder["add"](env.get("_1999"), env.get("_585"), {"label":"/model/layers.12/post_attention_layernorm/SkipLayerNorm_add_1649"}));
  env.set("_2001", builder["sqrt"](env.get("_2000"), {"label":"/model/layers.12/post_attention_layernorm/SkipLayerNorm_sqrt_1650"}));
  env.set("_2002", builder["div"](env.get("_1997"), env.get("_2001"), {"label":"/model/layers.12/post_attention_layernorm/SkipLayerNorm_div_1651"}));
  env.set("_2004", builder["mul"](env.get("_2003"), env.get("_2002"), {"label":"/model/layers.12/post_attention_layernorm/SkipLayerNorm_mul_1652"}));
  env.set("_2012", builder["matmul"](env.get("_2004"), env.get("_2011"), {"label":"/model/layers.12/mlp/gate_proj/MatMul_Q4_matmul_1657"}));
  env.set("_2013", builder["sigmoid"](env.get("_2012"), {"label":"/model/layers.12/mlp/act_fn/Sigmoid_1658"}));
  env.set("_2014", builder["mul"](env.get("_2012"), env.get("_2013"), {"label":"/model/layers.12/mlp/act_fn/Mul_1659"}));
  env.set("_276", builder.dequantizeLinear(env.get("_273"), env.get("_274"), env.get("_275"), {"axis":2,"blockSize":32,"label":"/model/layers.12/mlp/up_proj/MatMul_Q4_dequantizeLinear_114"}));
  env.set("_277", builder.reshape(env.get("_276"), [5632,2048]));
  env.set("_278", builder["transpose"](env.get("_277"), {"label":"/model/layers.12/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_116","permutation":[1,0]}));
  env.set("_2005", builder["matmul"](env.get("_2004"), env.get("_278"), {"label":"/model/layers.12/mlp/up_proj/MatMul_Q4_matmul_1653"}));
  env.set("_2015", builder["mul"](env.get("_2014"), env.get("_2005"), {"label":"/model/layers.12/mlp/Mul_1660"}));
  env.set("_270", builder.dequantizeLinear(env.get("_267"), env.get("_268"), env.get("_269"), {"axis":2,"blockSize":32,"label":"/model/layers.12/mlp/down_proj/MatMul_Q4_dequantizeLinear_111"}));
  env.set("_271", builder.reshape(env.get("_270"), [2048,5632]));
  env.set("_272", builder["transpose"](env.get("_271"), {"label":"/model/layers.12/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_113","permutation":[1,0]}));
  env.set("_2016", builder["matmul"](env.get("_2015"), env.get("_272"), {"label":"/model/layers.12/mlp/down_proj/MatMul_Q4_matmul_1661"}));
  env.set("_2017", builder["add"](env.get("_1997"), env.get("_2016"), {"label":"/model/layers.13/input_layernorm/SkipLayerNorm_add_skip_1662"}));
  env.set("_2018", builder["pow"](env.get("_2017"), env.get("_582"), {"label":"/model/layers.13/input_layernorm/SkipLayerNorm_pow_1663"}));
  env.set("_2019", builder["reduceMean"](env.get("_2018"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.13/input_layernorm/SkipLayerNorm_reduceMean_1664"}));
  env.set("_2020", builder["add"](env.get("_2019"), env.get("_585"), {"label":"/model/layers.13/input_layernorm/SkipLayerNorm_add_1665"}));
  env.set("_2021", builder["sqrt"](env.get("_2020"), {"label":"/model/layers.13/input_layernorm/SkipLayerNorm_sqrt_1666"}));
  env.set("_2022", builder["div"](env.get("_2017"), env.get("_2021"), {"label":"/model/layers.13/input_layernorm/SkipLayerNorm_div_1667"}));
  env.set("_2024", builder["mul"](env.get("_2023"), env.get("_2022"), {"label":"/model/layers.13/input_layernorm/SkipLayerNorm_mul_1668"}));
  env.set("_2080", builder["matmul"](env.get("_2024"), env.get("_2079"), {"label":"/model/layers.13/attn/q_proj/MatMul_Q4_matmul_1728"}));
  env.set("_2081", builder.reshape(env.get("_2080"), [1,sequence_length,32,64]));
  env.set("_2082", builder.reshape(env.get("_2081"), [1,sequence_length,32,2,32]));
  env.set("_2092", builder["mul"](env.get("_2082"), env.get("_2091"), {"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_mul_cos_1739"}));
  env.set("_2093", builder.reshape(env.get("_2092"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_2082"), 2, {"axis":3,"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_split_partial_input0_1731"});
    env.set("_2083", tmp[0]);
    env.set("_2084", tmp[1]);
  }
  env.set("_2085", builder.concat([env.get("_2084"), env.get("_2083")], 3, {"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1732"}));
  env.set("Inserted_915", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1723","maxValue":2047,"minValue":-2048}));
  env.set("_2072", builder["gather"](env.get("_621"), env.get("Inserted_915"), {"axis":0,"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_gather_sin_1722"}));
  env.set("_2073", builder.reshape(env.get("_2072"), [1,sequence_length,1,1,32]));
  env.set("_2086", builder["mul"](env.get("_2085"), env.get("_2073"), {"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_mul_sin_1733"}));
  env.set("_2088", builder["mul"](env.get("_2086"), env.get("_2087"), {"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_mul_sign_1734"}));
  env.set("_2089", builder.reshape(env.get("_2088"), [1,sequence_length,32,64]));
  env.set("_2094", builder["add"](env.get("_2093"), env.get("_2089"), {"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1741"}));
  env.set("_2095", builder.reshape(env.get("_2094"), [1,sequence_length,2048]));
  env.set("_2096", builder.reshape(env.get("_2095"), [1,sequence_length,32,64]));
  env.set("_2097", builder["transpose"](env.get("_2096"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/query/transpose_1744","permutation":[0,2,1,3]}));
  env.set("Inserted_876", builder.cast(env.get("_598"), "uint8"));
  env.set("_2027", builder["where"](env.get("Inserted_876"), env.get("_599"), env.get("_597"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/scatter/where_1671"}));
  env.set("_2028", builder["add"](env.get("_601"), env.get("_2027"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/right_constant/add_1673"}));
  env.set("_2029", builder.concat([env.get("_603"), env.get("_2028")], 2, {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1674"}));
  env.set("_2030", builder.reshape(env.get("_2029"), [1,sequence_length,4,3]));
  env.set("Inserted_907", builder.cast(env.get("_2030"), "int64"));
  env.set("Inserted_909", builder["max"](env.get("Inserted_907"), env.get("Inserted_908"), {"label":"Inserted_Max_1716"}));
  env.set("Inserted_911", builder["min"](env.get("Inserted_909"), env.get("Inserted_910"), {"label":"Inserted_Min_1717"}));
  env.set("Inserted_900", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1707","maxValue":2047,"minValue":-2048}));
  env.set("_2060", builder["gather"](env.get("_641"), env.get("Inserted_900"), {"axis":0,"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_gather_cos_1706"}));
  env.set("_2061", builder.reshape(env.get("_2060"), [1,sequence_length,1,1,32]));
  env.set("_2047", builder.dequantizeLinear(env.get("_2044"), env.get("_2045"), env.get("_2046"), {"axis":2,"blockSize":32,"label":"/model/layers.13/attn/k_proj/MatMul_Q4_dequantizeLinear_1695"}));
  env.set("_2048", builder.reshape(env.get("_2047"), [256,2048]));
  env.set("_2049", builder["transpose"](env.get("_2048"), {"label":"/model/layers.13/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1697","permutation":[1,0]}));
  env.set("_2050", builder["matmul"](env.get("_2024"), env.get("_2049"), {"label":"/model/layers.13/attn/k_proj/MatMul_Q4_matmul_1698"}));
  env.set("_2051", builder.reshape(env.get("_2050"), [1,sequence_length,4,64]));
  env.set("_2052", builder.reshape(env.get("_2051"), [1,sequence_length,4,2,32]));
  env.set("_2062", builder["mul"](env.get("_2052"), env.get("_2061"), {"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_mul_cos_1709"}));
  env.set("_2063", builder.reshape(env.get("_2062"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_2052"), 2, {"axis":3,"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_split_partial_input0_1701"});
    env.set("_2053", tmp[0]);
    env.set("_2054", tmp[1]);
  }
  env.set("_2055", builder.concat([env.get("_2054"), env.get("_2053")], 3, {"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1702"}));
  env.set("Inserted_891", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1693","maxValue":2047,"minValue":-2048}));
  env.set("_2042", builder["gather"](env.get("_621"), env.get("Inserted_891"), {"axis":0,"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_gather_sin_1692"}));
  env.set("_2043", builder.reshape(env.get("_2042"), [1,sequence_length,1,1,32]));
  env.set("_2056", builder["mul"](env.get("_2055"), env.get("_2043"), {"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_mul_sin_1703"}));
  env.set("_2058", builder["mul"](env.get("_2056"), env.get("_2057"), {"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_mul_sign_1704"}));
  env.set("_2059", builder.reshape(env.get("_2058"), [1,sequence_length,4,64]));
  env.set("_2064", builder["add"](env.get("_2063"), env.get("_2059"), {"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1711"}));
  env.set("_2065", builder.reshape(env.get("_2064"), [1,sequence_length,256]));
  env.set("_2066", builder.reshape(env.get("_2065"), [1,sequence_length,4,64]));
  env.set("present_13_key_26", builder["scatterND"](env.get("past_key_values_13_key_2067"), env.get("Inserted_911"), env.get("_2066"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1714"}));
  env.set("_2068", builder.reshape(env.get("present_13_key_26"), [1,4,1,past_sequence_length,64]));
  env.set("_2069", builder.expand(env.get("_2068"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/true_present_key/expand_1719"}));
  env.set("_2070", builder.reshape(env.get("_2069"), [1,32,past_sequence_length,64]));
  env.set("_2071", builder["transpose"](env.get("_2070"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/present_key/transpose_1721","permutation":[0,1,3,2]}));
  env.set("_2098", builder["matmul"](env.get("_2097"), env.get("_2071"), {"label":"/model/layers.13/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1745"}));
  env.set("_2099", builder["mul"](env.get("_2098"), env.get("_681"), {"label":"/model/layers.13/attn/GroupQueryAttention_/Attention/qkv/div_1746"}));
  env.set("_2038", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1686"}));
  env.set("_2039", builder.cumulativeSum(env.get("_2038"), 3, {"exclusive":true,"label":"/model/layers.13/attn/GroupQueryAttention_range_of_mask_shape_1687"}));
  env.set("_2035", builder["add"](env.get("_610"), env.get("_2027"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/attn_mask/add_1683"}));
  env.set("_2036", builder.expand(env.get("_2035"), [past_sequence_length,sequence_length], {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/expand_neq_right_1684"}));
  env.set("_2037", builder["transpose"](env.get("_2036"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/neq_right/transpose_1685","permutation":[1,0]}));
  env.set("Inserted_889", builder["lesser"](env.get("_2039"), env.get("_2037"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1688"}));
  env.set("_2040", builder.cast(env.get("Inserted_889"), "uint8"));
  env.set("Inserted_890", builder.cast(env.get("_2040"), "uint8"));
  env.set("_2041", builder["where"](env.get("Inserted_890"), env.get("_618"), env.get("_619"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/attn_mask/where_1690"}));
  env.set("_2100", builder["add"](env.get("_2099"), env.get("_2041"), {"label":"/model/layers.13/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1747"}));
  env.set("_2101", builder["softmax"](env.get("_2100"), 3));
  env.set("Inserted_878", builder.cast(env.get("_2030"), "int64"));
  env.set("Inserted_880", builder["max"](env.get("Inserted_878"), env.get("Inserted_879"), {"label":"Inserted_Max_1678"}));
  env.set("Inserted_882", builder["min"](env.get("Inserted_880"), env.get("Inserted_881"), {"label":"Inserted_Min_1679"}));
  env.set("_264", builder.dequantizeLinear(env.get("_261"), env.get("_262"), env.get("_263"), {"axis":2,"blockSize":32,"label":"/model/layers.13/attn/v_proj/MatMul_Q4_dequantizeLinear_108"}));
  env.set("_265", builder.reshape(env.get("_264"), [256,2048]));
  env.set("_266", builder["transpose"](env.get("_265"), {"label":"/model/layers.13/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_110","permutation":[1,0]}));
  env.set("_2025", builder["matmul"](env.get("_2024"), env.get("_266"), {"label":"/model/layers.13/attn/v_proj/MatMul_Q4_matmul_1669"}));
  env.set("_2026", builder.reshape(env.get("_2025"), [1,sequence_length,4,64]));
  env.set("present_13_value_27", builder["scatterND"](env.get("past_key_values_13_value_2031"), env.get("Inserted_882"), env.get("_2026"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1676"}));
  env.set("_2032", builder.reshape(env.get("present_13_value_27"), [1,4,1,past_sequence_length,64]));
  env.set("_2033", builder.expand(env.get("_2032"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/true_present_value/expand_1681"}));
  env.set("_2034", builder.reshape(env.get("_2033"), [1,32,past_sequence_length,64]));
  env.set("_2102", builder["matmul"](env.get("_2101"), env.get("_2034"), {"label":"/model/layers.13/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1749"}));
  env.set("_2103", builder["transpose"](env.get("_2102"), {"label":"/model/layers.13/attn/GroupQueryAttention_/Attention/qkv/transpose_1750","permutation":[0,2,1,3]}));
  env.set("_2104", builder.reshape(env.get("_2103"), [1,sequence_length,2048]));
  env.set("_258", builder.dequantizeLinear(env.get("_255"), env.get("_256"), env.get("_257"), {"axis":2,"blockSize":32,"label":"/model/layers.13/attn/o_proj/MatMul_Q4_dequantizeLinear_105"}));
  env.set("_259", builder.reshape(env.get("_258"), [2048,2048]));
  env.set("_260", builder["transpose"](env.get("_259"), {"label":"/model/layers.13/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_107","permutation":[1,0]}));
  env.set("_2105", builder["matmul"](env.get("_2104"), env.get("_260"), {"label":"/model/layers.13/attn/o_proj/MatMul_Q4_matmul_1752"}));
  env.set("_2106", builder["add"](env.get("_2017"), env.get("_2105"), {"label":"/model/layers.13/post_attention_layernorm/SkipLayerNorm_add_skip_1753"}));
  env.set("_2107", builder["pow"](env.get("_2106"), env.get("_582"), {"label":"/model/layers.13/post_attention_layernorm/SkipLayerNorm_pow_1754"}));
  env.set("_2108", builder["reduceMean"](env.get("_2107"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.13/post_attention_layernorm/SkipLayerNorm_reduceMean_1755"}));
  env.set("_2109", builder["add"](env.get("_2108"), env.get("_585"), {"label":"/model/layers.13/post_attention_layernorm/SkipLayerNorm_add_1756"}));
  env.set("_2110", builder["sqrt"](env.get("_2109"), {"label":"/model/layers.13/post_attention_layernorm/SkipLayerNorm_sqrt_1757"}));
  env.set("_2111", builder["div"](env.get("_2106"), env.get("_2110"), {"label":"/model/layers.13/post_attention_layernorm/SkipLayerNorm_div_1758"}));
  env.set("_2113", builder["mul"](env.get("_2112"), env.get("_2111"), {"label":"/model/layers.13/post_attention_layernorm/SkipLayerNorm_mul_1759"}));
  env.set("_2121", builder["matmul"](env.get("_2113"), env.get("_2120"), {"label":"/model/layers.13/mlp/gate_proj/MatMul_Q4_matmul_1764"}));
  env.set("_2122", builder["sigmoid"](env.get("_2121"), {"label":"/model/layers.13/mlp/act_fn/Sigmoid_1765"}));
  env.set("_2123", builder["mul"](env.get("_2121"), env.get("_2122"), {"label":"/model/layers.13/mlp/act_fn/Mul_1766"}));
  env.set("_252", builder.dequantizeLinear(env.get("_249"), env.get("_250"), env.get("_251"), {"axis":2,"blockSize":32,"label":"/model/layers.13/mlp/up_proj/MatMul_Q4_dequantizeLinear_102"}));
  env.set("_253", builder.reshape(env.get("_252"), [5632,2048]));
  env.set("_254", builder["transpose"](env.get("_253"), {"label":"/model/layers.13/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_104","permutation":[1,0]}));
  env.set("_2114", builder["matmul"](env.get("_2113"), env.get("_254"), {"label":"/model/layers.13/mlp/up_proj/MatMul_Q4_matmul_1760"}));
  env.set("_2124", builder["mul"](env.get("_2123"), env.get("_2114"), {"label":"/model/layers.13/mlp/Mul_1767"}));
  env.set("_246", builder.dequantizeLinear(env.get("_243"), env.get("_244"), env.get("_245"), {"axis":2,"blockSize":32,"label":"/model/layers.13/mlp/down_proj/MatMul_Q4_dequantizeLinear_99"}));
  env.set("_247", builder.reshape(env.get("_246"), [2048,5632]));
  env.set("_248", builder["transpose"](env.get("_247"), {"label":"/model/layers.13/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_101","permutation":[1,0]}));
  env.set("_2125", builder["matmul"](env.get("_2124"), env.get("_248"), {"label":"/model/layers.13/mlp/down_proj/MatMul_Q4_matmul_1768"}));
  env.set("_2126", builder["add"](env.get("_2106"), env.get("_2125"), {"label":"/model/layers.14/input_layernorm/SkipLayerNorm_add_skip_1769"}));
  env.set("_2127", builder["pow"](env.get("_2126"), env.get("_582"), {"label":"/model/layers.14/input_layernorm/SkipLayerNorm_pow_1770"}));
  env.set("_2128", builder["reduceMean"](env.get("_2127"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.14/input_layernorm/SkipLayerNorm_reduceMean_1771"}));
  env.set("_2129", builder["add"](env.get("_2128"), env.get("_585"), {"label":"/model/layers.14/input_layernorm/SkipLayerNorm_add_1772"}));
  env.set("_2130", builder["sqrt"](env.get("_2129"), {"label":"/model/layers.14/input_layernorm/SkipLayerNorm_sqrt_1773"}));
  env.set("_2131", builder["div"](env.get("_2126"), env.get("_2130"), {"label":"/model/layers.14/input_layernorm/SkipLayerNorm_div_1774"}));
  env.set("_2133", builder["mul"](env.get("_2132"), env.get("_2131"), {"label":"/model/layers.14/input_layernorm/SkipLayerNorm_mul_1775"}));
  env.set("_2189", builder["matmul"](env.get("_2133"), env.get("_2188"), {"label":"/model/layers.14/attn/q_proj/MatMul_Q4_matmul_1835"}));
  env.set("_2190", builder.reshape(env.get("_2189"), [1,sequence_length,32,64]));
  env.set("_2191", builder.reshape(env.get("_2190"), [1,sequence_length,32,2,32]));
  env.set("_2201", builder["mul"](env.get("_2191"), env.get("_2200"), {"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_mul_cos_1846"}));
  env.set("_2202", builder.reshape(env.get("_2201"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_2191"), 2, {"axis":3,"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_split_partial_input0_1838"});
    env.set("_2192", tmp[0]);
    env.set("_2193", tmp[1]);
  }
  env.set("_2194", builder.concat([env.get("_2193"), env.get("_2192")], 3, {"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1839"}));
  env.set("Inserted_975", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1830","maxValue":2047,"minValue":-2048}));
  env.set("_2181", builder["gather"](env.get("_621"), env.get("Inserted_975"), {"axis":0,"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_gather_sin_1829"}));
  env.set("_2182", builder.reshape(env.get("_2181"), [1,sequence_length,1,1,32]));
  env.set("_2195", builder["mul"](env.get("_2194"), env.get("_2182"), {"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_mul_sin_1840"}));
  env.set("_2197", builder["mul"](env.get("_2195"), env.get("_2196"), {"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_mul_sign_1841"}));
  env.set("_2198", builder.reshape(env.get("_2197"), [1,sequence_length,32,64]));
  env.set("_2203", builder["add"](env.get("_2202"), env.get("_2198"), {"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1848"}));
  env.set("_2204", builder.reshape(env.get("_2203"), [1,sequence_length,2048]));
  env.set("_2205", builder.reshape(env.get("_2204"), [1,sequence_length,32,64]));
  env.set("_2206", builder["transpose"](env.get("_2205"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/query/transpose_1851","permutation":[0,2,1,3]}));
  env.set("Inserted_936", builder.cast(env.get("_598"), "uint8"));
  env.set("_2136", builder["where"](env.get("Inserted_936"), env.get("_599"), env.get("_597"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/scatter/where_1778"}));
  env.set("_2137", builder["add"](env.get("_601"), env.get("_2136"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/right_constant/add_1780"}));
  env.set("_2138", builder.concat([env.get("_603"), env.get("_2137")], 2, {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1781"}));
  env.set("_2139", builder.reshape(env.get("_2138"), [1,sequence_length,4,3]));
  env.set("Inserted_967", builder.cast(env.get("_2139"), "int64"));
  env.set("Inserted_969", builder["max"](env.get("Inserted_967"), env.get("Inserted_968"), {"label":"Inserted_Max_1823"}));
  env.set("Inserted_971", builder["min"](env.get("Inserted_969"), env.get("Inserted_970"), {"label":"Inserted_Min_1824"}));
  env.set("Inserted_960", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1814","maxValue":2047,"minValue":-2048}));
  env.set("_2169", builder["gather"](env.get("_641"), env.get("Inserted_960"), {"axis":0,"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_gather_cos_1813"}));
  env.set("_2170", builder.reshape(env.get("_2169"), [1,sequence_length,1,1,32]));
  env.set("_2156", builder.dequantizeLinear(env.get("_2153"), env.get("_2154"), env.get("_2155"), {"axis":2,"blockSize":32,"label":"/model/layers.14/attn/k_proj/MatMul_Q4_dequantizeLinear_1802"}));
  env.set("_2157", builder.reshape(env.get("_2156"), [256,2048]));
  env.set("_2158", builder["transpose"](env.get("_2157"), {"label":"/model/layers.14/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1804","permutation":[1,0]}));
  env.set("_2159", builder["matmul"](env.get("_2133"), env.get("_2158"), {"label":"/model/layers.14/attn/k_proj/MatMul_Q4_matmul_1805"}));
  env.set("_2160", builder.reshape(env.get("_2159"), [1,sequence_length,4,64]));
  env.set("_2161", builder.reshape(env.get("_2160"), [1,sequence_length,4,2,32]));
  env.set("_2171", builder["mul"](env.get("_2161"), env.get("_2170"), {"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_mul_cos_1816"}));
  env.set("_2172", builder.reshape(env.get("_2171"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_2161"), 2, {"axis":3,"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_split_partial_input0_1808"});
    env.set("_2162", tmp[0]);
    env.set("_2163", tmp[1]);
  }
  env.set("_2164", builder.concat([env.get("_2163"), env.get("_2162")], 3, {"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1809"}));
  env.set("Inserted_951", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1800","maxValue":2047,"minValue":-2048}));
  env.set("_2151", builder["gather"](env.get("_621"), env.get("Inserted_951"), {"axis":0,"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_gather_sin_1799"}));
  env.set("_2152", builder.reshape(env.get("_2151"), [1,sequence_length,1,1,32]));
  env.set("_2165", builder["mul"](env.get("_2164"), env.get("_2152"), {"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_mul_sin_1810"}));
  env.set("_2167", builder["mul"](env.get("_2165"), env.get("_2166"), {"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_mul_sign_1811"}));
  env.set("_2168", builder.reshape(env.get("_2167"), [1,sequence_length,4,64]));
  env.set("_2173", builder["add"](env.get("_2172"), env.get("_2168"), {"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1818"}));
  env.set("_2174", builder.reshape(env.get("_2173"), [1,sequence_length,256]));
  env.set("_2175", builder.reshape(env.get("_2174"), [1,sequence_length,4,64]));
  env.set("present_14_key_28", builder["scatterND"](env.get("past_key_values_14_key_2176"), env.get("Inserted_971"), env.get("_2175"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1821"}));
  env.set("_2177", builder.reshape(env.get("present_14_key_28"), [1,4,1,past_sequence_length,64]));
  env.set("_2178", builder.expand(env.get("_2177"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/true_present_key/expand_1826"}));
  env.set("_2179", builder.reshape(env.get("_2178"), [1,32,past_sequence_length,64]));
  env.set("_2180", builder["transpose"](env.get("_2179"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/present_key/transpose_1828","permutation":[0,1,3,2]}));
  env.set("_2207", builder["matmul"](env.get("_2206"), env.get("_2180"), {"label":"/model/layers.14/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1852"}));
  env.set("_2208", builder["mul"](env.get("_2207"), env.get("_681"), {"label":"/model/layers.14/attn/GroupQueryAttention_/Attention/qkv/div_1853"}));
  env.set("_2147", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1793"}));
  env.set("_2148", builder.cumulativeSum(env.get("_2147"), 3, {"exclusive":true,"label":"/model/layers.14/attn/GroupQueryAttention_range_of_mask_shape_1794"}));
  env.set("_2144", builder["add"](env.get("_610"), env.get("_2136"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/attn_mask/add_1790"}));
  env.set("_2145", builder.expand(env.get("_2144"), [past_sequence_length,sequence_length], {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/expand_neq_right_1791"}));
  env.set("_2146", builder["transpose"](env.get("_2145"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/neq_right/transpose_1792","permutation":[1,0]}));
  env.set("Inserted_949", builder["lesser"](env.get("_2148"), env.get("_2146"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1795"}));
  env.set("_2149", builder.cast(env.get("Inserted_949"), "uint8"));
  env.set("Inserted_950", builder.cast(env.get("_2149"), "uint8"));
  env.set("_2150", builder["where"](env.get("Inserted_950"), env.get("_618"), env.get("_619"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/attn_mask/where_1797"}));
  env.set("_2209", builder["add"](env.get("_2208"), env.get("_2150"), {"label":"/model/layers.14/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1854"}));
  env.set("_2210", builder["softmax"](env.get("_2209"), 3));
  env.set("Inserted_938", builder.cast(env.get("_2139"), "int64"));
  env.set("Inserted_940", builder["max"](env.get("Inserted_938"), env.get("Inserted_939"), {"label":"Inserted_Max_1785"}));
  env.set("Inserted_942", builder["min"](env.get("Inserted_940"), env.get("Inserted_941"), {"label":"Inserted_Min_1786"}));
  env.set("_240", builder.dequantizeLinear(env.get("_237"), env.get("_238"), env.get("_239"), {"axis":2,"blockSize":32,"label":"/model/layers.14/attn/v_proj/MatMul_Q4_dequantizeLinear_96"}));
  env.set("_241", builder.reshape(env.get("_240"), [256,2048]));
  env.set("_242", builder["transpose"](env.get("_241"), {"label":"/model/layers.14/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_98","permutation":[1,0]}));
  env.set("_2134", builder["matmul"](env.get("_2133"), env.get("_242"), {"label":"/model/layers.14/attn/v_proj/MatMul_Q4_matmul_1776"}));
  env.set("_2135", builder.reshape(env.get("_2134"), [1,sequence_length,4,64]));
  env.set("present_14_value_29", builder["scatterND"](env.get("past_key_values_14_value_2140"), env.get("Inserted_942"), env.get("_2135"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1783"}));
  env.set("_2141", builder.reshape(env.get("present_14_value_29"), [1,4,1,past_sequence_length,64]));
  env.set("_2142", builder.expand(env.get("_2141"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/true_present_value/expand_1788"}));
  env.set("_2143", builder.reshape(env.get("_2142"), [1,32,past_sequence_length,64]));
  env.set("_2211", builder["matmul"](env.get("_2210"), env.get("_2143"), {"label":"/model/layers.14/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1856"}));
  env.set("_2212", builder["transpose"](env.get("_2211"), {"label":"/model/layers.14/attn/GroupQueryAttention_/Attention/qkv/transpose_1857","permutation":[0,2,1,3]}));
  env.set("_2213", builder.reshape(env.get("_2212"), [1,sequence_length,2048]));
  env.set("_234", builder.dequantizeLinear(env.get("_231"), env.get("_232"), env.get("_233"), {"axis":2,"blockSize":32,"label":"/model/layers.14/attn/o_proj/MatMul_Q4_dequantizeLinear_93"}));
  env.set("_235", builder.reshape(env.get("_234"), [2048,2048]));
  env.set("_236", builder["transpose"](env.get("_235"), {"label":"/model/layers.14/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_95","permutation":[1,0]}));
  env.set("_2214", builder["matmul"](env.get("_2213"), env.get("_236"), {"label":"/model/layers.14/attn/o_proj/MatMul_Q4_matmul_1859"}));
  env.set("_2215", builder["add"](env.get("_2126"), env.get("_2214"), {"label":"/model/layers.14/post_attention_layernorm/SkipLayerNorm_add_skip_1860"}));
  env.set("_2216", builder["pow"](env.get("_2215"), env.get("_582"), {"label":"/model/layers.14/post_attention_layernorm/SkipLayerNorm_pow_1861"}));
  env.set("_2217", builder["reduceMean"](env.get("_2216"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.14/post_attention_layernorm/SkipLayerNorm_reduceMean_1862"}));
  env.set("_2218", builder["add"](env.get("_2217"), env.get("_585"), {"label":"/model/layers.14/post_attention_layernorm/SkipLayerNorm_add_1863"}));
  env.set("_2219", builder["sqrt"](env.get("_2218"), {"label":"/model/layers.14/post_attention_layernorm/SkipLayerNorm_sqrt_1864"}));
  env.set("_2220", builder["div"](env.get("_2215"), env.get("_2219"), {"label":"/model/layers.14/post_attention_layernorm/SkipLayerNorm_div_1865"}));
  env.set("_2222", builder["mul"](env.get("_2221"), env.get("_2220"), {"label":"/model/layers.14/post_attention_layernorm/SkipLayerNorm_mul_1866"}));
  env.set("_2230", builder["matmul"](env.get("_2222"), env.get("_2229"), {"label":"/model/layers.14/mlp/gate_proj/MatMul_Q4_matmul_1871"}));
  env.set("_2231", builder["sigmoid"](env.get("_2230"), {"label":"/model/layers.14/mlp/act_fn/Sigmoid_1872"}));
  env.set("_2232", builder["mul"](env.get("_2230"), env.get("_2231"), {"label":"/model/layers.14/mlp/act_fn/Mul_1873"}));
  env.set("_228", builder.dequantizeLinear(env.get("_225"), env.get("_226"), env.get("_227"), {"axis":2,"blockSize":32,"label":"/model/layers.14/mlp/up_proj/MatMul_Q4_dequantizeLinear_90"}));
  env.set("_229", builder.reshape(env.get("_228"), [5632,2048]));
  env.set("_230", builder["transpose"](env.get("_229"), {"label":"/model/layers.14/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_92","permutation":[1,0]}));
  env.set("_2223", builder["matmul"](env.get("_2222"), env.get("_230"), {"label":"/model/layers.14/mlp/up_proj/MatMul_Q4_matmul_1867"}));
  env.set("_2233", builder["mul"](env.get("_2232"), env.get("_2223"), {"label":"/model/layers.14/mlp/Mul_1874"}));
  env.set("_222", builder.dequantizeLinear(env.get("_219"), env.get("_220"), env.get("_221"), {"axis":2,"blockSize":32,"label":"/model/layers.14/mlp/down_proj/MatMul_Q4_dequantizeLinear_87"}));
  env.set("_223", builder.reshape(env.get("_222"), [2048,5632]));
  env.set("_224", builder["transpose"](env.get("_223"), {"label":"/model/layers.14/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_89","permutation":[1,0]}));
  env.set("_2234", builder["matmul"](env.get("_2233"), env.get("_224"), {"label":"/model/layers.14/mlp/down_proj/MatMul_Q4_matmul_1875"}));
  env.set("_2235", builder["add"](env.get("_2215"), env.get("_2234"), {"label":"/model/layers.15/input_layernorm/SkipLayerNorm_add_skip_1876"}));
  env.set("_2236", builder["pow"](env.get("_2235"), env.get("_582"), {"label":"/model/layers.15/input_layernorm/SkipLayerNorm_pow_1877"}));
  env.set("_2237", builder["reduceMean"](env.get("_2236"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.15/input_layernorm/SkipLayerNorm_reduceMean_1878"}));
  env.set("_2238", builder["add"](env.get("_2237"), env.get("_585"), {"label":"/model/layers.15/input_layernorm/SkipLayerNorm_add_1879"}));
  env.set("_2239", builder["sqrt"](env.get("_2238"), {"label":"/model/layers.15/input_layernorm/SkipLayerNorm_sqrt_1880"}));
  env.set("_2240", builder["div"](env.get("_2235"), env.get("_2239"), {"label":"/model/layers.15/input_layernorm/SkipLayerNorm_div_1881"}));
  env.set("_2242", builder["mul"](env.get("_2241"), env.get("_2240"), {"label":"/model/layers.15/input_layernorm/SkipLayerNorm_mul_1882"}));
  env.set("_2298", builder["matmul"](env.get("_2242"), env.get("_2297"), {"label":"/model/layers.15/attn/q_proj/MatMul_Q4_matmul_1942"}));
  env.set("_2299", builder.reshape(env.get("_2298"), [1,sequence_length,32,64]));
  env.set("_2300", builder.reshape(env.get("_2299"), [1,sequence_length,32,2,32]));
  env.set("_2310", builder["mul"](env.get("_2300"), env.get("_2309"), {"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_mul_cos_1953"}));
  env.set("_2311", builder.reshape(env.get("_2310"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_2300"), 2, {"axis":3,"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_split_partial_input0_1945"});
    env.set("_2301", tmp[0]);
    env.set("_2302", tmp[1]);
  }
  env.set("_2303", builder.concat([env.get("_2302"), env.get("_2301")], 3, {"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1946"}));
  env.set("Inserted_1035", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1937","maxValue":2047,"minValue":-2048}));
  env.set("_2290", builder["gather"](env.get("_621"), env.get("Inserted_1035"), {"axis":0,"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_gather_sin_1936"}));
  env.set("_2291", builder.reshape(env.get("_2290"), [1,sequence_length,1,1,32]));
  env.set("_2304", builder["mul"](env.get("_2303"), env.get("_2291"), {"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_mul_sin_1947"}));
  env.set("_2306", builder["mul"](env.get("_2304"), env.get("_2305"), {"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_mul_sign_1948"}));
  env.set("_2307", builder.reshape(env.get("_2306"), [1,sequence_length,32,64]));
  env.set("_2312", builder["add"](env.get("_2311"), env.get("_2307"), {"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1955"}));
  env.set("_2313", builder.reshape(env.get("_2312"), [1,sequence_length,2048]));
  env.set("_2314", builder.reshape(env.get("_2313"), [1,sequence_length,32,64]));
  env.set("_2315", builder["transpose"](env.get("_2314"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/query/transpose_1958","permutation":[0,2,1,3]}));
  env.set("Inserted_996", builder.cast(env.get("_598"), "uint8"));
  env.set("_2245", builder["where"](env.get("Inserted_996"), env.get("_599"), env.get("_597"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/scatter/where_1885"}));
  env.set("_2246", builder["add"](env.get("_601"), env.get("_2245"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/right_constant/add_1887"}));
  env.set("_2247", builder.concat([env.get("_603"), env.get("_2246")], 2, {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1888"}));
  env.set("_2248", builder.reshape(env.get("_2247"), [1,sequence_length,4,3]));
  env.set("Inserted_1027", builder.cast(env.get("_2248"), "int64"));
  env.set("Inserted_1029", builder["max"](env.get("Inserted_1027"), env.get("Inserted_1028"), {"label":"Inserted_Max_1930"}));
  env.set("Inserted_1031", builder["min"](env.get("Inserted_1029"), env.get("Inserted_1030"), {"label":"Inserted_Min_1931"}));
  env.set("Inserted_1020", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1921","maxValue":2047,"minValue":-2048}));
  env.set("_2278", builder["gather"](env.get("_641"), env.get("Inserted_1020"), {"axis":0,"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_gather_cos_1920"}));
  env.set("_2279", builder.reshape(env.get("_2278"), [1,sequence_length,1,1,32]));
  env.set("_2265", builder.dequantizeLinear(env.get("_2262"), env.get("_2263"), env.get("_2264"), {"axis":2,"blockSize":32,"label":"/model/layers.15/attn/k_proj/MatMul_Q4_dequantizeLinear_1909"}));
  env.set("_2266", builder.reshape(env.get("_2265"), [256,2048]));
  env.set("_2267", builder["transpose"](env.get("_2266"), {"label":"/model/layers.15/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1911","permutation":[1,0]}));
  env.set("_2268", builder["matmul"](env.get("_2242"), env.get("_2267"), {"label":"/model/layers.15/attn/k_proj/MatMul_Q4_matmul_1912"}));
  env.set("_2269", builder.reshape(env.get("_2268"), [1,sequence_length,4,64]));
  env.set("_2270", builder.reshape(env.get("_2269"), [1,sequence_length,4,2,32]));
  env.set("_2280", builder["mul"](env.get("_2270"), env.get("_2279"), {"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_mul_cos_1923"}));
  env.set("_2281", builder.reshape(env.get("_2280"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_2270"), 2, {"axis":3,"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_split_partial_input0_1915"});
    env.set("_2271", tmp[0]);
    env.set("_2272", tmp[1]);
  }
  env.set("_2273", builder.concat([env.get("_2272"), env.get("_2271")], 3, {"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1916"}));
  env.set("Inserted_1011", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_1907","maxValue":2047,"minValue":-2048}));
  env.set("_2260", builder["gather"](env.get("_621"), env.get("Inserted_1011"), {"axis":0,"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_gather_sin_1906"}));
  env.set("_2261", builder.reshape(env.get("_2260"), [1,sequence_length,1,1,32]));
  env.set("_2274", builder["mul"](env.get("_2273"), env.get("_2261"), {"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_mul_sin_1917"}));
  env.set("_2276", builder["mul"](env.get("_2274"), env.get("_2275"), {"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_mul_sign_1918"}));
  env.set("_2277", builder.reshape(env.get("_2276"), [1,sequence_length,4,64]));
  env.set("_2282", builder["add"](env.get("_2281"), env.get("_2277"), {"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1925"}));
  env.set("_2283", builder.reshape(env.get("_2282"), [1,sequence_length,256]));
  env.set("_2284", builder.reshape(env.get("_2283"), [1,sequence_length,4,64]));
  env.set("present_15_key_30", builder["scatterND"](env.get("past_key_values_15_key_2285"), env.get("Inserted_1031"), env.get("_2284"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1928"}));
  env.set("_2286", builder.reshape(env.get("present_15_key_30"), [1,4,1,past_sequence_length,64]));
  env.set("_2287", builder.expand(env.get("_2286"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/true_present_key/expand_1933"}));
  env.set("_2288", builder.reshape(env.get("_2287"), [1,32,past_sequence_length,64]));
  env.set("_2289", builder["transpose"](env.get("_2288"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/present_key/transpose_1935","permutation":[0,1,3,2]}));
  env.set("_2316", builder["matmul"](env.get("_2315"), env.get("_2289"), {"label":"/model/layers.15/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1959"}));
  env.set("_2317", builder["mul"](env.get("_2316"), env.get("_681"), {"label":"/model/layers.15/attn/GroupQueryAttention_/Attention/qkv/div_1960"}));
  env.set("_2256", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1900"}));
  env.set("_2257", builder.cumulativeSum(env.get("_2256"), 3, {"exclusive":true,"label":"/model/layers.15/attn/GroupQueryAttention_range_of_mask_shape_1901"}));
  env.set("_2253", builder["add"](env.get("_610"), env.get("_2245"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/attn_mask/add_1897"}));
  env.set("_2254", builder.expand(env.get("_2253"), [past_sequence_length,sequence_length], {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/expand_neq_right_1898"}));
  env.set("_2255", builder["transpose"](env.get("_2254"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/neq_right/transpose_1899","permutation":[1,0]}));
  env.set("Inserted_1009", builder["lesser"](env.get("_2257"), env.get("_2255"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1902"}));
  env.set("_2258", builder.cast(env.get("Inserted_1009"), "uint8"));
  env.set("Inserted_1010", builder.cast(env.get("_2258"), "uint8"));
  env.set("_2259", builder["where"](env.get("Inserted_1010"), env.get("_618"), env.get("_619"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/attn_mask/where_1904"}));
  env.set("_2318", builder["add"](env.get("_2317"), env.get("_2259"), {"label":"/model/layers.15/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1961"}));
  env.set("_2319", builder["softmax"](env.get("_2318"), 3));
  env.set("Inserted_998", builder.cast(env.get("_2248"), "int64"));
  env.set("Inserted_1000", builder["max"](env.get("Inserted_998"), env.get("Inserted_999"), {"label":"Inserted_Max_1892"}));
  env.set("Inserted_1002", builder["min"](env.get("Inserted_1000"), env.get("Inserted_1001"), {"label":"Inserted_Min_1893"}));
  env.set("_216", builder.dequantizeLinear(env.get("_213"), env.get("_214"), env.get("_215"), {"axis":2,"blockSize":32,"label":"/model/layers.15/attn/v_proj/MatMul_Q4_dequantizeLinear_84"}));
  env.set("_217", builder.reshape(env.get("_216"), [256,2048]));
  env.set("_218", builder["transpose"](env.get("_217"), {"label":"/model/layers.15/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_86","permutation":[1,0]}));
  env.set("_2243", builder["matmul"](env.get("_2242"), env.get("_218"), {"label":"/model/layers.15/attn/v_proj/MatMul_Q4_matmul_1883"}));
  env.set("_2244", builder.reshape(env.get("_2243"), [1,sequence_length,4,64]));
  env.set("present_15_value_31", builder["scatterND"](env.get("past_key_values_15_value_2249"), env.get("Inserted_1002"), env.get("_2244"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1890"}));
  env.set("_2250", builder.reshape(env.get("present_15_value_31"), [1,4,1,past_sequence_length,64]));
  env.set("_2251", builder.expand(env.get("_2250"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/true_present_value/expand_1895"}));
  env.set("_2252", builder.reshape(env.get("_2251"), [1,32,past_sequence_length,64]));
  env.set("_2320", builder["matmul"](env.get("_2319"), env.get("_2252"), {"label":"/model/layers.15/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1963"}));
  env.set("_2321", builder["transpose"](env.get("_2320"), {"label":"/model/layers.15/attn/GroupQueryAttention_/Attention/qkv/transpose_1964","permutation":[0,2,1,3]}));
  env.set("_2322", builder.reshape(env.get("_2321"), [1,sequence_length,2048]));
  env.set("_210", builder.dequantizeLinear(env.get("_207"), env.get("_208"), env.get("_209"), {"axis":2,"blockSize":32,"label":"/model/layers.15/attn/o_proj/MatMul_Q4_dequantizeLinear_81"}));
  env.set("_211", builder.reshape(env.get("_210"), [2048,2048]));
  env.set("_212", builder["transpose"](env.get("_211"), {"label":"/model/layers.15/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_83","permutation":[1,0]}));
  env.set("_2323", builder["matmul"](env.get("_2322"), env.get("_212"), {"label":"/model/layers.15/attn/o_proj/MatMul_Q4_matmul_1966"}));
  env.set("_2324", builder["add"](env.get("_2235"), env.get("_2323"), {"label":"/model/layers.15/post_attention_layernorm/SkipLayerNorm_add_skip_1967"}));
  env.set("_2325", builder["pow"](env.get("_2324"), env.get("_582"), {"label":"/model/layers.15/post_attention_layernorm/SkipLayerNorm_pow_1968"}));
  env.set("_2326", builder["reduceMean"](env.get("_2325"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.15/post_attention_layernorm/SkipLayerNorm_reduceMean_1969"}));
  env.set("_2327", builder["add"](env.get("_2326"), env.get("_585"), {"label":"/model/layers.15/post_attention_layernorm/SkipLayerNorm_add_1970"}));
  env.set("_2328", builder["sqrt"](env.get("_2327"), {"label":"/model/layers.15/post_attention_layernorm/SkipLayerNorm_sqrt_1971"}));
  env.set("_2329", builder["div"](env.get("_2324"), env.get("_2328"), {"label":"/model/layers.15/post_attention_layernorm/SkipLayerNorm_div_1972"}));
  env.set("_2331", builder["mul"](env.get("_2330"), env.get("_2329"), {"label":"/model/layers.15/post_attention_layernorm/SkipLayerNorm_mul_1973"}));
  env.set("_2339", builder["matmul"](env.get("_2331"), env.get("_2338"), {"label":"/model/layers.15/mlp/gate_proj/MatMul_Q4_matmul_1978"}));
  env.set("_2340", builder["sigmoid"](env.get("_2339"), {"label":"/model/layers.15/mlp/act_fn/Sigmoid_1979"}));
  env.set("_2341", builder["mul"](env.get("_2339"), env.get("_2340"), {"label":"/model/layers.15/mlp/act_fn/Mul_1980"}));
  env.set("_204", builder.dequantizeLinear(env.get("_201"), env.get("_202"), env.get("_203"), {"axis":2,"blockSize":32,"label":"/model/layers.15/mlp/up_proj/MatMul_Q4_dequantizeLinear_78"}));
  env.set("_205", builder.reshape(env.get("_204"), [5632,2048]));
  env.set("_206", builder["transpose"](env.get("_205"), {"label":"/model/layers.15/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_80","permutation":[1,0]}));
  env.set("_2332", builder["matmul"](env.get("_2331"), env.get("_206"), {"label":"/model/layers.15/mlp/up_proj/MatMul_Q4_matmul_1974"}));
  env.set("_2342", builder["mul"](env.get("_2341"), env.get("_2332"), {"label":"/model/layers.15/mlp/Mul_1981"}));
  env.set("_198", builder.dequantizeLinear(env.get("_195"), env.get("_196"), env.get("_197"), {"axis":2,"blockSize":32,"label":"/model/layers.15/mlp/down_proj/MatMul_Q4_dequantizeLinear_75"}));
  env.set("_199", builder.reshape(env.get("_198"), [2048,5632]));
  env.set("_200", builder["transpose"](env.get("_199"), {"label":"/model/layers.15/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_77","permutation":[1,0]}));
  env.set("_2343", builder["matmul"](env.get("_2342"), env.get("_200"), {"label":"/model/layers.15/mlp/down_proj/MatMul_Q4_matmul_1982"}));
  env.set("_2344", builder["add"](env.get("_2324"), env.get("_2343"), {"label":"/model/layers.16/input_layernorm/SkipLayerNorm_add_skip_1983"}));
  env.set("_2345", builder["pow"](env.get("_2344"), env.get("_582"), {"label":"/model/layers.16/input_layernorm/SkipLayerNorm_pow_1984"}));
  env.set("_2346", builder["reduceMean"](env.get("_2345"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.16/input_layernorm/SkipLayerNorm_reduceMean_1985"}));
  env.set("_2347", builder["add"](env.get("_2346"), env.get("_585"), {"label":"/model/layers.16/input_layernorm/SkipLayerNorm_add_1986"}));
  env.set("_2348", builder["sqrt"](env.get("_2347"), {"label":"/model/layers.16/input_layernorm/SkipLayerNorm_sqrt_1987"}));
  env.set("_2349", builder["div"](env.get("_2344"), env.get("_2348"), {"label":"/model/layers.16/input_layernorm/SkipLayerNorm_div_1988"}));
  env.set("_2351", builder["mul"](env.get("_2350"), env.get("_2349"), {"label":"/model/layers.16/input_layernorm/SkipLayerNorm_mul_1989"}));
  env.set("_2407", builder["matmul"](env.get("_2351"), env.get("_2406"), {"label":"/model/layers.16/attn/q_proj/MatMul_Q4_matmul_2049"}));
  env.set("_2408", builder.reshape(env.get("_2407"), [1,sequence_length,32,64]));
  env.set("_2409", builder.reshape(env.get("_2408"), [1,sequence_length,32,2,32]));
  env.set("_2419", builder["mul"](env.get("_2409"), env.get("_2418"), {"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_mul_cos_2060"}));
  env.set("_2420", builder.reshape(env.get("_2419"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_2409"), 2, {"axis":3,"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_split_partial_input0_2052"});
    env.set("_2410", tmp[0]);
    env.set("_2411", tmp[1]);
  }
  env.set("_2412", builder.concat([env.get("_2411"), env.get("_2410")], 3, {"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_concat_partial_input0_2053"}));
  env.set("Inserted_1095", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2044","maxValue":2047,"minValue":-2048}));
  env.set("_2399", builder["gather"](env.get("_621"), env.get("Inserted_1095"), {"axis":0,"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_gather_sin_2043"}));
  env.set("_2400", builder.reshape(env.get("_2399"), [1,sequence_length,1,1,32]));
  env.set("_2413", builder["mul"](env.get("_2412"), env.get("_2400"), {"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_mul_sin_2054"}));
  env.set("_2415", builder["mul"](env.get("_2413"), env.get("_2414"), {"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_mul_sign_2055"}));
  env.set("_2416", builder.reshape(env.get("_2415"), [1,sequence_length,32,64]));
  env.set("_2421", builder["add"](env.get("_2420"), env.get("_2416"), {"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_2062"}));
  env.set("_2422", builder.reshape(env.get("_2421"), [1,sequence_length,2048]));
  env.set("_2423", builder.reshape(env.get("_2422"), [1,sequence_length,32,64]));
  env.set("_2424", builder["transpose"](env.get("_2423"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/query/transpose_2065","permutation":[0,2,1,3]}));
  env.set("Inserted_1056", builder.cast(env.get("_598"), "uint8"));
  env.set("_2354", builder["where"](env.get("Inserted_1056"), env.get("_599"), env.get("_597"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/scatter/where_1992"}));
  env.set("_2355", builder["add"](env.get("_601"), env.get("_2354"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/right_constant/add_1994"}));
  env.set("_2356", builder.concat([env.get("_603"), env.get("_2355")], 2, {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1995"}));
  env.set("_2357", builder.reshape(env.get("_2356"), [1,sequence_length,4,3]));
  env.set("Inserted_1087", builder.cast(env.get("_2357"), "int64"));
  env.set("Inserted_1089", builder["max"](env.get("Inserted_1087"), env.get("Inserted_1088"), {"label":"Inserted_Max_2037"}));
  env.set("Inserted_1091", builder["min"](env.get("Inserted_1089"), env.get("Inserted_1090"), {"label":"Inserted_Min_2038"}));
  env.set("Inserted_1080", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2028","maxValue":2047,"minValue":-2048}));
  env.set("_2387", builder["gather"](env.get("_641"), env.get("Inserted_1080"), {"axis":0,"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_gather_cos_2027"}));
  env.set("_2388", builder.reshape(env.get("_2387"), [1,sequence_length,1,1,32]));
  env.set("_2374", builder.dequantizeLinear(env.get("_2371"), env.get("_2372"), env.get("_2373"), {"axis":2,"blockSize":32,"label":"/model/layers.16/attn/k_proj/MatMul_Q4_dequantizeLinear_2016"}));
  env.set("_2375", builder.reshape(env.get("_2374"), [256,2048]));
  env.set("_2376", builder["transpose"](env.get("_2375"), {"label":"/model/layers.16/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_2018","permutation":[1,0]}));
  env.set("_2377", builder["matmul"](env.get("_2351"), env.get("_2376"), {"label":"/model/layers.16/attn/k_proj/MatMul_Q4_matmul_2019"}));
  env.set("_2378", builder.reshape(env.get("_2377"), [1,sequence_length,4,64]));
  env.set("_2379", builder.reshape(env.get("_2378"), [1,sequence_length,4,2,32]));
  env.set("_2389", builder["mul"](env.get("_2379"), env.get("_2388"), {"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_mul_cos_2030"}));
  env.set("_2390", builder.reshape(env.get("_2389"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_2379"), 2, {"axis":3,"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_split_partial_input0_2022"});
    env.set("_2380", tmp[0]);
    env.set("_2381", tmp[1]);
  }
  env.set("_2382", builder.concat([env.get("_2381"), env.get("_2380")], 3, {"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_concat_partial_input0_2023"}));
  env.set("Inserted_1071", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2014","maxValue":2047,"minValue":-2048}));
  env.set("_2369", builder["gather"](env.get("_621"), env.get("Inserted_1071"), {"axis":0,"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_gather_sin_2013"}));
  env.set("_2370", builder.reshape(env.get("_2369"), [1,sequence_length,1,1,32]));
  env.set("_2383", builder["mul"](env.get("_2382"), env.get("_2370"), {"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_mul_sin_2024"}));
  env.set("_2385", builder["mul"](env.get("_2383"), env.get("_2384"), {"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_mul_sign_2025"}));
  env.set("_2386", builder.reshape(env.get("_2385"), [1,sequence_length,4,64]));
  env.set("_2391", builder["add"](env.get("_2390"), env.get("_2386"), {"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_2032"}));
  env.set("_2392", builder.reshape(env.get("_2391"), [1,sequence_length,256]));
  env.set("_2393", builder.reshape(env.get("_2392"), [1,sequence_length,4,64]));
  env.set("present_16_key_32", builder["scatterND"](env.get("past_key_values_16_key_2394"), env.get("Inserted_1091"), env.get("_2393"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/present_key/ScatterND_2035"}));
  env.set("_2395", builder.reshape(env.get("present_16_key_32"), [1,4,1,past_sequence_length,64]));
  env.set("_2396", builder.expand(env.get("_2395"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/true_present_key/expand_2040"}));
  env.set("_2397", builder.reshape(env.get("_2396"), [1,32,past_sequence_length,64]));
  env.set("_2398", builder["transpose"](env.get("_2397"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/present_key/transpose_2042","permutation":[0,1,3,2]}));
  env.set("_2425", builder["matmul"](env.get("_2424"), env.get("_2398"), {"label":"/model/layers.16/attn/GroupQueryAttention_/Attention/qkv/matmul_1_2066"}));
  env.set("_2426", builder["mul"](env.get("_2425"), env.get("_681"), {"label":"/model/layers.16/attn/GroupQueryAttention_/Attention/qkv/div_2067"}));
  env.set("_2365", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_2007"}));
  env.set("_2366", builder.cumulativeSum(env.get("_2365"), 3, {"exclusive":true,"label":"/model/layers.16/attn/GroupQueryAttention_range_of_mask_shape_2008"}));
  env.set("_2362", builder["add"](env.get("_610"), env.get("_2354"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/attn_mask/add_2004"}));
  env.set("_2363", builder.expand(env.get("_2362"), [past_sequence_length,sequence_length], {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/expand_neq_right_2005"}));
  env.set("_2364", builder["transpose"](env.get("_2363"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/neq_right/transpose_2006","permutation":[1,0]}));
  env.set("Inserted_1069", builder["lesser"](env.get("_2366"), env.get("_2364"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_2009"}));
  env.set("_2367", builder.cast(env.get("Inserted_1069"), "uint8"));
  env.set("Inserted_1070", builder.cast(env.get("_2367"), "uint8"));
  env.set("_2368", builder["where"](env.get("Inserted_1070"), env.get("_618"), env.get("_619"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/attn_mask/where_2011"}));
  env.set("_2427", builder["add"](env.get("_2426"), env.get("_2368"), {"label":"/model/layers.16/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_2068"}));
  env.set("_2428", builder["softmax"](env.get("_2427"), 3));
  env.set("Inserted_1058", builder.cast(env.get("_2357"), "int64"));
  env.set("Inserted_1060", builder["max"](env.get("Inserted_1058"), env.get("Inserted_1059"), {"label":"Inserted_Max_1999"}));
  env.set("Inserted_1062", builder["min"](env.get("Inserted_1060"), env.get("Inserted_1061"), {"label":"Inserted_Min_2000"}));
  env.set("_192", builder.dequantizeLinear(env.get("_189"), env.get("_190"), env.get("_191"), {"axis":2,"blockSize":32,"label":"/model/layers.16/attn/v_proj/MatMul_Q4_dequantizeLinear_72"}));
  env.set("_193", builder.reshape(env.get("_192"), [256,2048]));
  env.set("_194", builder["transpose"](env.get("_193"), {"label":"/model/layers.16/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_74","permutation":[1,0]}));
  env.set("_2352", builder["matmul"](env.get("_2351"), env.get("_194"), {"label":"/model/layers.16/attn/v_proj/MatMul_Q4_matmul_1990"}));
  env.set("_2353", builder.reshape(env.get("_2352"), [1,sequence_length,4,64]));
  env.set("present_16_value_33", builder["scatterND"](env.get("past_key_values_16_value_2358"), env.get("Inserted_1062"), env.get("_2353"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1997"}));
  env.set("_2359", builder.reshape(env.get("present_16_value_33"), [1,4,1,past_sequence_length,64]));
  env.set("_2360", builder.expand(env.get("_2359"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/true_present_value/expand_2002"}));
  env.set("_2361", builder.reshape(env.get("_2360"), [1,32,past_sequence_length,64]));
  env.set("_2429", builder["matmul"](env.get("_2428"), env.get("_2361"), {"label":"/model/layers.16/attn/GroupQueryAttention_/Attention/qkv/matmul_2_2070"}));
  env.set("_2430", builder["transpose"](env.get("_2429"), {"label":"/model/layers.16/attn/GroupQueryAttention_/Attention/qkv/transpose_2071","permutation":[0,2,1,3]}));
  env.set("_2431", builder.reshape(env.get("_2430"), [1,sequence_length,2048]));
  env.set("_186", builder.dequantizeLinear(env.get("_183"), env.get("_184"), env.get("_185"), {"axis":2,"blockSize":32,"label":"/model/layers.16/attn/o_proj/MatMul_Q4_dequantizeLinear_69"}));
  env.set("_187", builder.reshape(env.get("_186"), [2048,2048]));
  env.set("_188", builder["transpose"](env.get("_187"), {"label":"/model/layers.16/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_71","permutation":[1,0]}));
  env.set("_2432", builder["matmul"](env.get("_2431"), env.get("_188"), {"label":"/model/layers.16/attn/o_proj/MatMul_Q4_matmul_2073"}));
  env.set("_2433", builder["add"](env.get("_2344"), env.get("_2432"), {"label":"/model/layers.16/post_attention_layernorm/SkipLayerNorm_add_skip_2074"}));
  env.set("_2434", builder["pow"](env.get("_2433"), env.get("_582"), {"label":"/model/layers.16/post_attention_layernorm/SkipLayerNorm_pow_2075"}));
  env.set("_2435", builder["reduceMean"](env.get("_2434"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.16/post_attention_layernorm/SkipLayerNorm_reduceMean_2076"}));
  env.set("_2436", builder["add"](env.get("_2435"), env.get("_585"), {"label":"/model/layers.16/post_attention_layernorm/SkipLayerNorm_add_2077"}));
  env.set("_2437", builder["sqrt"](env.get("_2436"), {"label":"/model/layers.16/post_attention_layernorm/SkipLayerNorm_sqrt_2078"}));
  env.set("_2438", builder["div"](env.get("_2433"), env.get("_2437"), {"label":"/model/layers.16/post_attention_layernorm/SkipLayerNorm_div_2079"}));
  env.set("_2440", builder["mul"](env.get("_2439"), env.get("_2438"), {"label":"/model/layers.16/post_attention_layernorm/SkipLayerNorm_mul_2080"}));
  env.set("_2448", builder["matmul"](env.get("_2440"), env.get("_2447"), {"label":"/model/layers.16/mlp/gate_proj/MatMul_Q4_matmul_2085"}));
  env.set("_2449", builder["sigmoid"](env.get("_2448"), {"label":"/model/layers.16/mlp/act_fn/Sigmoid_2086"}));
  env.set("_2450", builder["mul"](env.get("_2448"), env.get("_2449"), {"label":"/model/layers.16/mlp/act_fn/Mul_2087"}));
  env.set("_180", builder.dequantizeLinear(env.get("_177"), env.get("_178"), env.get("_179"), {"axis":2,"blockSize":32,"label":"/model/layers.16/mlp/up_proj/MatMul_Q4_dequantizeLinear_66"}));
  env.set("_181", builder.reshape(env.get("_180"), [5632,2048]));
  env.set("_182", builder["transpose"](env.get("_181"), {"label":"/model/layers.16/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_68","permutation":[1,0]}));
  env.set("_2441", builder["matmul"](env.get("_2440"), env.get("_182"), {"label":"/model/layers.16/mlp/up_proj/MatMul_Q4_matmul_2081"}));
  env.set("_2451", builder["mul"](env.get("_2450"), env.get("_2441"), {"label":"/model/layers.16/mlp/Mul_2088"}));
  env.set("_174", builder.dequantizeLinear(env.get("_171"), env.get("_172"), env.get("_173"), {"axis":2,"blockSize":32,"label":"/model/layers.16/mlp/down_proj/MatMul_Q4_dequantizeLinear_63"}));
  env.set("_175", builder.reshape(env.get("_174"), [2048,5632]));
  env.set("_176", builder["transpose"](env.get("_175"), {"label":"/model/layers.16/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_65","permutation":[1,0]}));
  env.set("_2452", builder["matmul"](env.get("_2451"), env.get("_176"), {"label":"/model/layers.16/mlp/down_proj/MatMul_Q4_matmul_2089"}));
  env.set("_2453", builder["add"](env.get("_2433"), env.get("_2452"), {"label":"/model/layers.17/input_layernorm/SkipLayerNorm_add_skip_2090"}));
  env.set("_2454", builder["pow"](env.get("_2453"), env.get("_582"), {"label":"/model/layers.17/input_layernorm/SkipLayerNorm_pow_2091"}));
  env.set("_2455", builder["reduceMean"](env.get("_2454"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.17/input_layernorm/SkipLayerNorm_reduceMean_2092"}));
  env.set("_2456", builder["add"](env.get("_2455"), env.get("_585"), {"label":"/model/layers.17/input_layernorm/SkipLayerNorm_add_2093"}));
  env.set("_2457", builder["sqrt"](env.get("_2456"), {"label":"/model/layers.17/input_layernorm/SkipLayerNorm_sqrt_2094"}));
  env.set("_2458", builder["div"](env.get("_2453"), env.get("_2457"), {"label":"/model/layers.17/input_layernorm/SkipLayerNorm_div_2095"}));
  env.set("_2460", builder["mul"](env.get("_2459"), env.get("_2458"), {"label":"/model/layers.17/input_layernorm/SkipLayerNorm_mul_2096"}));
  env.set("_2516", builder["matmul"](env.get("_2460"), env.get("_2515"), {"label":"/model/layers.17/attn/q_proj/MatMul_Q4_matmul_2156"}));
  env.set("_2517", builder.reshape(env.get("_2516"), [1,sequence_length,32,64]));
  env.set("_2518", builder.reshape(env.get("_2517"), [1,sequence_length,32,2,32]));
  env.set("_2528", builder["mul"](env.get("_2518"), env.get("_2527"), {"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_mul_cos_2167"}));
  env.set("_2529", builder.reshape(env.get("_2528"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_2518"), 2, {"axis":3,"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_split_partial_input0_2159"});
    env.set("_2519", tmp[0]);
    env.set("_2520", tmp[1]);
  }
  env.set("_2521", builder.concat([env.get("_2520"), env.get("_2519")], 3, {"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_concat_partial_input0_2160"}));
  env.set("Inserted_1155", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2151","maxValue":2047,"minValue":-2048}));
  env.set("_2508", builder["gather"](env.get("_621"), env.get("Inserted_1155"), {"axis":0,"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_gather_sin_2150"}));
  env.set("_2509", builder.reshape(env.get("_2508"), [1,sequence_length,1,1,32]));
  env.set("_2522", builder["mul"](env.get("_2521"), env.get("_2509"), {"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_mul_sin_2161"}));
  env.set("_2524", builder["mul"](env.get("_2522"), env.get("_2523"), {"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_mul_sign_2162"}));
  env.set("_2525", builder.reshape(env.get("_2524"), [1,sequence_length,32,64]));
  env.set("_2530", builder["add"](env.get("_2529"), env.get("_2525"), {"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_2169"}));
  env.set("_2531", builder.reshape(env.get("_2530"), [1,sequence_length,2048]));
  env.set("_2532", builder.reshape(env.get("_2531"), [1,sequence_length,32,64]));
  env.set("_2533", builder["transpose"](env.get("_2532"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/query/transpose_2172","permutation":[0,2,1,3]}));
  env.set("Inserted_1116", builder.cast(env.get("_598"), "uint8"));
  env.set("_2463", builder["where"](env.get("Inserted_1116"), env.get("_599"), env.get("_597"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/scatter/where_2099"}));
  env.set("_2464", builder["add"](env.get("_601"), env.get("_2463"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/right_constant/add_2101"}));
  env.set("_2465", builder.concat([env.get("_603"), env.get("_2464")], 2, {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_2102"}));
  env.set("_2466", builder.reshape(env.get("_2465"), [1,sequence_length,4,3]));
  env.set("Inserted_1147", builder.cast(env.get("_2466"), "int64"));
  env.set("Inserted_1149", builder["max"](env.get("Inserted_1147"), env.get("Inserted_1148"), {"label":"Inserted_Max_2144"}));
  env.set("Inserted_1151", builder["min"](env.get("Inserted_1149"), env.get("Inserted_1150"), {"label":"Inserted_Min_2145"}));
  env.set("Inserted_1140", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2135","maxValue":2047,"minValue":-2048}));
  env.set("_2496", builder["gather"](env.get("_641"), env.get("Inserted_1140"), {"axis":0,"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_gather_cos_2134"}));
  env.set("_2497", builder.reshape(env.get("_2496"), [1,sequence_length,1,1,32]));
  env.set("_2483", builder.dequantizeLinear(env.get("_2480"), env.get("_2481"), env.get("_2482"), {"axis":2,"blockSize":32,"label":"/model/layers.17/attn/k_proj/MatMul_Q4_dequantizeLinear_2123"}));
  env.set("_2484", builder.reshape(env.get("_2483"), [256,2048]));
  env.set("_2485", builder["transpose"](env.get("_2484"), {"label":"/model/layers.17/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_2125","permutation":[1,0]}));
  env.set("_2486", builder["matmul"](env.get("_2460"), env.get("_2485"), {"label":"/model/layers.17/attn/k_proj/MatMul_Q4_matmul_2126"}));
  env.set("_2487", builder.reshape(env.get("_2486"), [1,sequence_length,4,64]));
  env.set("_2488", builder.reshape(env.get("_2487"), [1,sequence_length,4,2,32]));
  env.set("_2498", builder["mul"](env.get("_2488"), env.get("_2497"), {"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_mul_cos_2137"}));
  env.set("_2499", builder.reshape(env.get("_2498"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_2488"), 2, {"axis":3,"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_split_partial_input0_2129"});
    env.set("_2489", tmp[0]);
    env.set("_2490", tmp[1]);
  }
  env.set("_2491", builder.concat([env.get("_2490"), env.get("_2489")], 3, {"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_concat_partial_input0_2130"}));
  env.set("Inserted_1131", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2121","maxValue":2047,"minValue":-2048}));
  env.set("_2478", builder["gather"](env.get("_621"), env.get("Inserted_1131"), {"axis":0,"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_gather_sin_2120"}));
  env.set("_2479", builder.reshape(env.get("_2478"), [1,sequence_length,1,1,32]));
  env.set("_2492", builder["mul"](env.get("_2491"), env.get("_2479"), {"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_mul_sin_2131"}));
  env.set("_2494", builder["mul"](env.get("_2492"), env.get("_2493"), {"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_mul_sign_2132"}));
  env.set("_2495", builder.reshape(env.get("_2494"), [1,sequence_length,4,64]));
  env.set("_2500", builder["add"](env.get("_2499"), env.get("_2495"), {"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_2139"}));
  env.set("_2501", builder.reshape(env.get("_2500"), [1,sequence_length,256]));
  env.set("_2502", builder.reshape(env.get("_2501"), [1,sequence_length,4,64]));
  env.set("present_17_key_34", builder["scatterND"](env.get("past_key_values_17_key_2503"), env.get("Inserted_1151"), env.get("_2502"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/present_key/ScatterND_2142"}));
  env.set("_2504", builder.reshape(env.get("present_17_key_34"), [1,4,1,past_sequence_length,64]));
  env.set("_2505", builder.expand(env.get("_2504"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/true_present_key/expand_2147"}));
  env.set("_2506", builder.reshape(env.get("_2505"), [1,32,past_sequence_length,64]));
  env.set("_2507", builder["transpose"](env.get("_2506"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/present_key/transpose_2149","permutation":[0,1,3,2]}));
  env.set("_2534", builder["matmul"](env.get("_2533"), env.get("_2507"), {"label":"/model/layers.17/attn/GroupQueryAttention_/Attention/qkv/matmul_1_2173"}));
  env.set("_2535", builder["mul"](env.get("_2534"), env.get("_681"), {"label":"/model/layers.17/attn/GroupQueryAttention_/Attention/qkv/div_2174"}));
  env.set("_2474", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_2114"}));
  env.set("_2475", builder.cumulativeSum(env.get("_2474"), 3, {"exclusive":true,"label":"/model/layers.17/attn/GroupQueryAttention_range_of_mask_shape_2115"}));
  env.set("_2471", builder["add"](env.get("_610"), env.get("_2463"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/attn_mask/add_2111"}));
  env.set("_2472", builder.expand(env.get("_2471"), [past_sequence_length,sequence_length], {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/expand_neq_right_2112"}));
  env.set("_2473", builder["transpose"](env.get("_2472"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/neq_right/transpose_2113","permutation":[1,0]}));
  env.set("Inserted_1129", builder["lesser"](env.get("_2475"), env.get("_2473"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_2116"}));
  env.set("_2476", builder.cast(env.get("Inserted_1129"), "uint8"));
  env.set("Inserted_1130", builder.cast(env.get("_2476"), "uint8"));
  env.set("_2477", builder["where"](env.get("Inserted_1130"), env.get("_618"), env.get("_619"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/attn_mask/where_2118"}));
  env.set("_2536", builder["add"](env.get("_2535"), env.get("_2477"), {"label":"/model/layers.17/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_2175"}));
  env.set("_2537", builder["softmax"](env.get("_2536"), 3));
  env.set("Inserted_1118", builder.cast(env.get("_2466"), "int64"));
  env.set("Inserted_1120", builder["max"](env.get("Inserted_1118"), env.get("Inserted_1119"), {"label":"Inserted_Max_2106"}));
  env.set("Inserted_1122", builder["min"](env.get("Inserted_1120"), env.get("Inserted_1121"), {"label":"Inserted_Min_2107"}));
  env.set("_168", builder.dequantizeLinear(env.get("_165"), env.get("_166"), env.get("_167"), {"axis":2,"blockSize":32,"label":"/model/layers.17/attn/v_proj/MatMul_Q4_dequantizeLinear_60"}));
  env.set("_169", builder.reshape(env.get("_168"), [256,2048]));
  env.set("_170", builder["transpose"](env.get("_169"), {"label":"/model/layers.17/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_62","permutation":[1,0]}));
  env.set("_2461", builder["matmul"](env.get("_2460"), env.get("_170"), {"label":"/model/layers.17/attn/v_proj/MatMul_Q4_matmul_2097"}));
  env.set("_2462", builder.reshape(env.get("_2461"), [1,sequence_length,4,64]));
  env.set("present_17_value_35", builder["scatterND"](env.get("past_key_values_17_value_2467"), env.get("Inserted_1122"), env.get("_2462"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/present_value/ScatterND_2104"}));
  env.set("_2468", builder.reshape(env.get("present_17_value_35"), [1,4,1,past_sequence_length,64]));
  env.set("_2469", builder.expand(env.get("_2468"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/true_present_value/expand_2109"}));
  env.set("_2470", builder.reshape(env.get("_2469"), [1,32,past_sequence_length,64]));
  env.set("_2538", builder["matmul"](env.get("_2537"), env.get("_2470"), {"label":"/model/layers.17/attn/GroupQueryAttention_/Attention/qkv/matmul_2_2177"}));
  env.set("_2539", builder["transpose"](env.get("_2538"), {"label":"/model/layers.17/attn/GroupQueryAttention_/Attention/qkv/transpose_2178","permutation":[0,2,1,3]}));
  env.set("_2540", builder.reshape(env.get("_2539"), [1,sequence_length,2048]));
  env.set("_162", builder.dequantizeLinear(env.get("_159"), env.get("_160"), env.get("_161"), {"axis":2,"blockSize":32,"label":"/model/layers.17/attn/o_proj/MatMul_Q4_dequantizeLinear_57"}));
  env.set("_163", builder.reshape(env.get("_162"), [2048,2048]));
  env.set("_164", builder["transpose"](env.get("_163"), {"label":"/model/layers.17/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_59","permutation":[1,0]}));
  env.set("_2541", builder["matmul"](env.get("_2540"), env.get("_164"), {"label":"/model/layers.17/attn/o_proj/MatMul_Q4_matmul_2180"}));
  env.set("_2542", builder["add"](env.get("_2453"), env.get("_2541"), {"label":"/model/layers.17/post_attention_layernorm/SkipLayerNorm_add_skip_2181"}));
  env.set("_2543", builder["pow"](env.get("_2542"), env.get("_582"), {"label":"/model/layers.17/post_attention_layernorm/SkipLayerNorm_pow_2182"}));
  env.set("_2544", builder["reduceMean"](env.get("_2543"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.17/post_attention_layernorm/SkipLayerNorm_reduceMean_2183"}));
  env.set("_2545", builder["add"](env.get("_2544"), env.get("_585"), {"label":"/model/layers.17/post_attention_layernorm/SkipLayerNorm_add_2184"}));
  env.set("_2546", builder["sqrt"](env.get("_2545"), {"label":"/model/layers.17/post_attention_layernorm/SkipLayerNorm_sqrt_2185"}));
  env.set("_2547", builder["div"](env.get("_2542"), env.get("_2546"), {"label":"/model/layers.17/post_attention_layernorm/SkipLayerNorm_div_2186"}));
  env.set("_2549", builder["mul"](env.get("_2548"), env.get("_2547"), {"label":"/model/layers.17/post_attention_layernorm/SkipLayerNorm_mul_2187"}));
  env.set("_2557", builder["matmul"](env.get("_2549"), env.get("_2556"), {"label":"/model/layers.17/mlp/gate_proj/MatMul_Q4_matmul_2192"}));
  env.set("_2558", builder["sigmoid"](env.get("_2557"), {"label":"/model/layers.17/mlp/act_fn/Sigmoid_2193"}));
  env.set("_2559", builder["mul"](env.get("_2557"), env.get("_2558"), {"label":"/model/layers.17/mlp/act_fn/Mul_2194"}));
  env.set("_156", builder.dequantizeLinear(env.get("_153"), env.get("_154"), env.get("_155"), {"axis":2,"blockSize":32,"label":"/model/layers.17/mlp/up_proj/MatMul_Q4_dequantizeLinear_54"}));
  env.set("_157", builder.reshape(env.get("_156"), [5632,2048]));
  env.set("_158", builder["transpose"](env.get("_157"), {"label":"/model/layers.17/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_56","permutation":[1,0]}));
  env.set("_2550", builder["matmul"](env.get("_2549"), env.get("_158"), {"label":"/model/layers.17/mlp/up_proj/MatMul_Q4_matmul_2188"}));
  env.set("_2560", builder["mul"](env.get("_2559"), env.get("_2550"), {"label":"/model/layers.17/mlp/Mul_2195"}));
  env.set("_150", builder.dequantizeLinear(env.get("_147"), env.get("_148"), env.get("_149"), {"axis":2,"blockSize":32,"label":"/model/layers.17/mlp/down_proj/MatMul_Q4_dequantizeLinear_51"}));
  env.set("_151", builder.reshape(env.get("_150"), [2048,5632]));
  env.set("_152", builder["transpose"](env.get("_151"), {"label":"/model/layers.17/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_53","permutation":[1,0]}));
  env.set("_2561", builder["matmul"](env.get("_2560"), env.get("_152"), {"label":"/model/layers.17/mlp/down_proj/MatMul_Q4_matmul_2196"}));
  env.set("_2562", builder["add"](env.get("_2542"), env.get("_2561"), {"label":"/model/layers.18/input_layernorm/SkipLayerNorm_add_skip_2197"}));
  env.set("_2563", builder["pow"](env.get("_2562"), env.get("_582"), {"label":"/model/layers.18/input_layernorm/SkipLayerNorm_pow_2198"}));
  env.set("_2564", builder["reduceMean"](env.get("_2563"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.18/input_layernorm/SkipLayerNorm_reduceMean_2199"}));
  env.set("_2565", builder["add"](env.get("_2564"), env.get("_585"), {"label":"/model/layers.18/input_layernorm/SkipLayerNorm_add_2200"}));
  env.set("_2566", builder["sqrt"](env.get("_2565"), {"label":"/model/layers.18/input_layernorm/SkipLayerNorm_sqrt_2201"}));
  env.set("_2567", builder["div"](env.get("_2562"), env.get("_2566"), {"label":"/model/layers.18/input_layernorm/SkipLayerNorm_div_2202"}));
  env.set("_2569", builder["mul"](env.get("_2568"), env.get("_2567"), {"label":"/model/layers.18/input_layernorm/SkipLayerNorm_mul_2203"}));
  env.set("_2625", builder["matmul"](env.get("_2569"), env.get("_2624"), {"label":"/model/layers.18/attn/q_proj/MatMul_Q4_matmul_2263"}));
  env.set("_2626", builder.reshape(env.get("_2625"), [1,sequence_length,32,64]));
  env.set("_2627", builder.reshape(env.get("_2626"), [1,sequence_length,32,2,32]));
  env.set("_2637", builder["mul"](env.get("_2627"), env.get("_2636"), {"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_mul_cos_2274"}));
  env.set("_2638", builder.reshape(env.get("_2637"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_2627"), 2, {"axis":3,"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_split_partial_input0_2266"});
    env.set("_2628", tmp[0]);
    env.set("_2629", tmp[1]);
  }
  env.set("_2630", builder.concat([env.get("_2629"), env.get("_2628")], 3, {"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_concat_partial_input0_2267"}));
  env.set("Inserted_1215", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2258","maxValue":2047,"minValue":-2048}));
  env.set("_2617", builder["gather"](env.get("_621"), env.get("Inserted_1215"), {"axis":0,"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_gather_sin_2257"}));
  env.set("_2618", builder.reshape(env.get("_2617"), [1,sequence_length,1,1,32]));
  env.set("_2631", builder["mul"](env.get("_2630"), env.get("_2618"), {"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_mul_sin_2268"}));
  env.set("_2633", builder["mul"](env.get("_2631"), env.get("_2632"), {"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_mul_sign_2269"}));
  env.set("_2634", builder.reshape(env.get("_2633"), [1,sequence_length,32,64]));
  env.set("_2639", builder["add"](env.get("_2638"), env.get("_2634"), {"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_2276"}));
  env.set("_2640", builder.reshape(env.get("_2639"), [1,sequence_length,2048]));
  env.set("_2641", builder.reshape(env.get("_2640"), [1,sequence_length,32,64]));
  env.set("_2642", builder["transpose"](env.get("_2641"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/query/transpose_2279","permutation":[0,2,1,3]}));
  env.set("Inserted_1176", builder.cast(env.get("_598"), "uint8"));
  env.set("_2572", builder["where"](env.get("Inserted_1176"), env.get("_599"), env.get("_597"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/scatter/where_2206"}));
  env.set("_2573", builder["add"](env.get("_601"), env.get("_2572"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/right_constant/add_2208"}));
  env.set("_2574", builder.concat([env.get("_603"), env.get("_2573")], 2, {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_2209"}));
  env.set("_2575", builder.reshape(env.get("_2574"), [1,sequence_length,4,3]));
  env.set("Inserted_1207", builder.cast(env.get("_2575"), "int64"));
  env.set("Inserted_1209", builder["max"](env.get("Inserted_1207"), env.get("Inserted_1208"), {"label":"Inserted_Max_2251"}));
  env.set("Inserted_1211", builder["min"](env.get("Inserted_1209"), env.get("Inserted_1210"), {"label":"Inserted_Min_2252"}));
  env.set("Inserted_1200", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2242","maxValue":2047,"minValue":-2048}));
  env.set("_2605", builder["gather"](env.get("_641"), env.get("Inserted_1200"), {"axis":0,"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_gather_cos_2241"}));
  env.set("_2606", builder.reshape(env.get("_2605"), [1,sequence_length,1,1,32]));
  env.set("_2592", builder.dequantizeLinear(env.get("_2589"), env.get("_2590"), env.get("_2591"), {"axis":2,"blockSize":32,"label":"/model/layers.18/attn/k_proj/MatMul_Q4_dequantizeLinear_2230"}));
  env.set("_2593", builder.reshape(env.get("_2592"), [256,2048]));
  env.set("_2594", builder["transpose"](env.get("_2593"), {"label":"/model/layers.18/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_2232","permutation":[1,0]}));
  env.set("_2595", builder["matmul"](env.get("_2569"), env.get("_2594"), {"label":"/model/layers.18/attn/k_proj/MatMul_Q4_matmul_2233"}));
  env.set("_2596", builder.reshape(env.get("_2595"), [1,sequence_length,4,64]));
  env.set("_2597", builder.reshape(env.get("_2596"), [1,sequence_length,4,2,32]));
  env.set("_2607", builder["mul"](env.get("_2597"), env.get("_2606"), {"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_mul_cos_2244"}));
  env.set("_2608", builder.reshape(env.get("_2607"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_2597"), 2, {"axis":3,"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_split_partial_input0_2236"});
    env.set("_2598", tmp[0]);
    env.set("_2599", tmp[1]);
  }
  env.set("_2600", builder.concat([env.get("_2599"), env.get("_2598")], 3, {"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_concat_partial_input0_2237"}));
  env.set("Inserted_1191", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2228","maxValue":2047,"minValue":-2048}));
  env.set("_2587", builder["gather"](env.get("_621"), env.get("Inserted_1191"), {"axis":0,"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_gather_sin_2227"}));
  env.set("_2588", builder.reshape(env.get("_2587"), [1,sequence_length,1,1,32]));
  env.set("_2601", builder["mul"](env.get("_2600"), env.get("_2588"), {"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_mul_sin_2238"}));
  env.set("_2603", builder["mul"](env.get("_2601"), env.get("_2602"), {"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_mul_sign_2239"}));
  env.set("_2604", builder.reshape(env.get("_2603"), [1,sequence_length,4,64]));
  env.set("_2609", builder["add"](env.get("_2608"), env.get("_2604"), {"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_2246"}));
  env.set("_2610", builder.reshape(env.get("_2609"), [1,sequence_length,256]));
  env.set("_2611", builder.reshape(env.get("_2610"), [1,sequence_length,4,64]));
  env.set("present_18_key_36", builder["scatterND"](env.get("past_key_values_18_key_2612"), env.get("Inserted_1211"), env.get("_2611"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/present_key/ScatterND_2249"}));
  env.set("_2613", builder.reshape(env.get("present_18_key_36"), [1,4,1,past_sequence_length,64]));
  env.set("_2614", builder.expand(env.get("_2613"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/true_present_key/expand_2254"}));
  env.set("_2615", builder.reshape(env.get("_2614"), [1,32,past_sequence_length,64]));
  env.set("_2616", builder["transpose"](env.get("_2615"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/present_key/transpose_2256","permutation":[0,1,3,2]}));
  env.set("_2643", builder["matmul"](env.get("_2642"), env.get("_2616"), {"label":"/model/layers.18/attn/GroupQueryAttention_/Attention/qkv/matmul_1_2280"}));
  env.set("_2644", builder["mul"](env.get("_2643"), env.get("_681"), {"label":"/model/layers.18/attn/GroupQueryAttention_/Attention/qkv/div_2281"}));
  env.set("_2583", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_2221"}));
  env.set("_2584", builder.cumulativeSum(env.get("_2583"), 3, {"exclusive":true,"label":"/model/layers.18/attn/GroupQueryAttention_range_of_mask_shape_2222"}));
  env.set("_2580", builder["add"](env.get("_610"), env.get("_2572"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/attn_mask/add_2218"}));
  env.set("_2581", builder.expand(env.get("_2580"), [past_sequence_length,sequence_length], {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/expand_neq_right_2219"}));
  env.set("_2582", builder["transpose"](env.get("_2581"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/neq_right/transpose_2220","permutation":[1,0]}));
  env.set("Inserted_1189", builder["lesser"](env.get("_2584"), env.get("_2582"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_2223"}));
  env.set("_2585", builder.cast(env.get("Inserted_1189"), "uint8"));
  env.set("Inserted_1190", builder.cast(env.get("_2585"), "uint8"));
  env.set("_2586", builder["where"](env.get("Inserted_1190"), env.get("_618"), env.get("_619"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/attn_mask/where_2225"}));
  env.set("_2645", builder["add"](env.get("_2644"), env.get("_2586"), {"label":"/model/layers.18/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_2282"}));
  env.set("_2646", builder["softmax"](env.get("_2645"), 3));
  env.set("Inserted_1178", builder.cast(env.get("_2575"), "int64"));
  env.set("Inserted_1180", builder["max"](env.get("Inserted_1178"), env.get("Inserted_1179"), {"label":"Inserted_Max_2213"}));
  env.set("Inserted_1182", builder["min"](env.get("Inserted_1180"), env.get("Inserted_1181"), {"label":"Inserted_Min_2214"}));
  env.set("_144", builder.dequantizeLinear(env.get("_141"), env.get("_142"), env.get("_143"), {"axis":2,"blockSize":32,"label":"/model/layers.18/attn/v_proj/MatMul_Q4_dequantizeLinear_48"}));
  env.set("_145", builder.reshape(env.get("_144"), [256,2048]));
  env.set("_146", builder["transpose"](env.get("_145"), {"label":"/model/layers.18/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_50","permutation":[1,0]}));
  env.set("_2570", builder["matmul"](env.get("_2569"), env.get("_146"), {"label":"/model/layers.18/attn/v_proj/MatMul_Q4_matmul_2204"}));
  env.set("_2571", builder.reshape(env.get("_2570"), [1,sequence_length,4,64]));
  env.set("present_18_value_37", builder["scatterND"](env.get("past_key_values_18_value_2576"), env.get("Inserted_1182"), env.get("_2571"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/present_value/ScatterND_2211"}));
  env.set("_2577", builder.reshape(env.get("present_18_value_37"), [1,4,1,past_sequence_length,64]));
  env.set("_2578", builder.expand(env.get("_2577"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/true_present_value/expand_2216"}));
  env.set("_2579", builder.reshape(env.get("_2578"), [1,32,past_sequence_length,64]));
  env.set("_2647", builder["matmul"](env.get("_2646"), env.get("_2579"), {"label":"/model/layers.18/attn/GroupQueryAttention_/Attention/qkv/matmul_2_2284"}));
  env.set("_2648", builder["transpose"](env.get("_2647"), {"label":"/model/layers.18/attn/GroupQueryAttention_/Attention/qkv/transpose_2285","permutation":[0,2,1,3]}));
  env.set("_2649", builder.reshape(env.get("_2648"), [1,sequence_length,2048]));
  env.set("_138", builder.dequantizeLinear(env.get("_135"), env.get("_136"), env.get("_137"), {"axis":2,"blockSize":32,"label":"/model/layers.18/attn/o_proj/MatMul_Q4_dequantizeLinear_45"}));
  env.set("_139", builder.reshape(env.get("_138"), [2048,2048]));
  env.set("_140", builder["transpose"](env.get("_139"), {"label":"/model/layers.18/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_47","permutation":[1,0]}));
  env.set("_2650", builder["matmul"](env.get("_2649"), env.get("_140"), {"label":"/model/layers.18/attn/o_proj/MatMul_Q4_matmul_2287"}));
  env.set("_2651", builder["add"](env.get("_2562"), env.get("_2650"), {"label":"/model/layers.18/post_attention_layernorm/SkipLayerNorm_add_skip_2288"}));
  env.set("_2652", builder["pow"](env.get("_2651"), env.get("_582"), {"label":"/model/layers.18/post_attention_layernorm/SkipLayerNorm_pow_2289"}));
  env.set("_2653", builder["reduceMean"](env.get("_2652"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.18/post_attention_layernorm/SkipLayerNorm_reduceMean_2290"}));
  env.set("_2654", builder["add"](env.get("_2653"), env.get("_585"), {"label":"/model/layers.18/post_attention_layernorm/SkipLayerNorm_add_2291"}));
  env.set("_2655", builder["sqrt"](env.get("_2654"), {"label":"/model/layers.18/post_attention_layernorm/SkipLayerNorm_sqrt_2292"}));
  env.set("_2656", builder["div"](env.get("_2651"), env.get("_2655"), {"label":"/model/layers.18/post_attention_layernorm/SkipLayerNorm_div_2293"}));
  env.set("_2658", builder["mul"](env.get("_2657"), env.get("_2656"), {"label":"/model/layers.18/post_attention_layernorm/SkipLayerNorm_mul_2294"}));
  env.set("_2666", builder["matmul"](env.get("_2658"), env.get("_2665"), {"label":"/model/layers.18/mlp/gate_proj/MatMul_Q4_matmul_2299"}));
  env.set("_2667", builder["sigmoid"](env.get("_2666"), {"label":"/model/layers.18/mlp/act_fn/Sigmoid_2300"}));
  env.set("_2668", builder["mul"](env.get("_2666"), env.get("_2667"), {"label":"/model/layers.18/mlp/act_fn/Mul_2301"}));
  env.set("_132", builder.dequantizeLinear(env.get("_129"), env.get("_130"), env.get("_131"), {"axis":2,"blockSize":32,"label":"/model/layers.18/mlp/up_proj/MatMul_Q4_dequantizeLinear_42"}));
  env.set("_133", builder.reshape(env.get("_132"), [5632,2048]));
  env.set("_134", builder["transpose"](env.get("_133"), {"label":"/model/layers.18/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_44","permutation":[1,0]}));
  env.set("_2659", builder["matmul"](env.get("_2658"), env.get("_134"), {"label":"/model/layers.18/mlp/up_proj/MatMul_Q4_matmul_2295"}));
  env.set("_2669", builder["mul"](env.get("_2668"), env.get("_2659"), {"label":"/model/layers.18/mlp/Mul_2302"}));
  env.set("_126", builder.dequantizeLinear(env.get("_123"), env.get("_124"), env.get("_125"), {"axis":2,"blockSize":32,"label":"/model/layers.18/mlp/down_proj/MatMul_Q4_dequantizeLinear_39"}));
  env.set("_127", builder.reshape(env.get("_126"), [2048,5632]));
  env.set("_128", builder["transpose"](env.get("_127"), {"label":"/model/layers.18/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_41","permutation":[1,0]}));
  env.set("_2670", builder["matmul"](env.get("_2669"), env.get("_128"), {"label":"/model/layers.18/mlp/down_proj/MatMul_Q4_matmul_2303"}));
  env.set("_2671", builder["add"](env.get("_2651"), env.get("_2670"), {"label":"/model/layers.19/input_layernorm/SkipLayerNorm_add_skip_2304"}));
  env.set("_2672", builder["pow"](env.get("_2671"), env.get("_582"), {"label":"/model/layers.19/input_layernorm/SkipLayerNorm_pow_2305"}));
  env.set("_2673", builder["reduceMean"](env.get("_2672"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.19/input_layernorm/SkipLayerNorm_reduceMean_2306"}));
  env.set("_2674", builder["add"](env.get("_2673"), env.get("_585"), {"label":"/model/layers.19/input_layernorm/SkipLayerNorm_add_2307"}));
  env.set("_2675", builder["sqrt"](env.get("_2674"), {"label":"/model/layers.19/input_layernorm/SkipLayerNorm_sqrt_2308"}));
  env.set("_2676", builder["div"](env.get("_2671"), env.get("_2675"), {"label":"/model/layers.19/input_layernorm/SkipLayerNorm_div_2309"}));
  env.set("_2678", builder["mul"](env.get("_2677"), env.get("_2676"), {"label":"/model/layers.19/input_layernorm/SkipLayerNorm_mul_2310"}));
  env.set("_2734", builder["matmul"](env.get("_2678"), env.get("_2733"), {"label":"/model/layers.19/attn/q_proj/MatMul_Q4_matmul_2370"}));
  env.set("_2735", builder.reshape(env.get("_2734"), [1,sequence_length,32,64]));
  env.set("_2736", builder.reshape(env.get("_2735"), [1,sequence_length,32,2,32]));
  env.set("_2746", builder["mul"](env.get("_2736"), env.get("_2745"), {"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_mul_cos_2381"}));
  env.set("_2747", builder.reshape(env.get("_2746"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_2736"), 2, {"axis":3,"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_split_partial_input0_2373"});
    env.set("_2737", tmp[0]);
    env.set("_2738", tmp[1]);
  }
  env.set("_2739", builder.concat([env.get("_2738"), env.get("_2737")], 3, {"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_concat_partial_input0_2374"}));
  env.set("Inserted_1275", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2365","maxValue":2047,"minValue":-2048}));
  env.set("_2726", builder["gather"](env.get("_621"), env.get("Inserted_1275"), {"axis":0,"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_gather_sin_2364"}));
  env.set("_2727", builder.reshape(env.get("_2726"), [1,sequence_length,1,1,32]));
  env.set("_2740", builder["mul"](env.get("_2739"), env.get("_2727"), {"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_mul_sin_2375"}));
  env.set("_2742", builder["mul"](env.get("_2740"), env.get("_2741"), {"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_mul_sign_2376"}));
  env.set("_2743", builder.reshape(env.get("_2742"), [1,sequence_length,32,64]));
  env.set("_2748", builder["add"](env.get("_2747"), env.get("_2743"), {"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_2383"}));
  env.set("_2749", builder.reshape(env.get("_2748"), [1,sequence_length,2048]));
  env.set("_2750", builder.reshape(env.get("_2749"), [1,sequence_length,32,64]));
  env.set("_2751", builder["transpose"](env.get("_2750"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/query/transpose_2386","permutation":[0,2,1,3]}));
  env.set("Inserted_1236", builder.cast(env.get("_598"), "uint8"));
  env.set("_2681", builder["where"](env.get("Inserted_1236"), env.get("_599"), env.get("_597"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/scatter/where_2313"}));
  env.set("_2682", builder["add"](env.get("_601"), env.get("_2681"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/right_constant/add_2315"}));
  env.set("_2683", builder.concat([env.get("_603"), env.get("_2682")], 2, {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_2316"}));
  env.set("_2684", builder.reshape(env.get("_2683"), [1,sequence_length,4,3]));
  env.set("Inserted_1267", builder.cast(env.get("_2684"), "int64"));
  env.set("Inserted_1269", builder["max"](env.get("Inserted_1267"), env.get("Inserted_1268"), {"label":"Inserted_Max_2358"}));
  env.set("Inserted_1271", builder["min"](env.get("Inserted_1269"), env.get("Inserted_1270"), {"label":"Inserted_Min_2359"}));
  env.set("Inserted_1260", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2349","maxValue":2047,"minValue":-2048}));
  env.set("_2714", builder["gather"](env.get("_641"), env.get("Inserted_1260"), {"axis":0,"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_gather_cos_2348"}));
  env.set("_2715", builder.reshape(env.get("_2714"), [1,sequence_length,1,1,32]));
  env.set("_2701", builder.dequantizeLinear(env.get("_2698"), env.get("_2699"), env.get("_2700"), {"axis":2,"blockSize":32,"label":"/model/layers.19/attn/k_proj/MatMul_Q4_dequantizeLinear_2337"}));
  env.set("_2702", builder.reshape(env.get("_2701"), [256,2048]));
  env.set("_2703", builder["transpose"](env.get("_2702"), {"label":"/model/layers.19/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_2339","permutation":[1,0]}));
  env.set("_2704", builder["matmul"](env.get("_2678"), env.get("_2703"), {"label":"/model/layers.19/attn/k_proj/MatMul_Q4_matmul_2340"}));
  env.set("_2705", builder.reshape(env.get("_2704"), [1,sequence_length,4,64]));
  env.set("_2706", builder.reshape(env.get("_2705"), [1,sequence_length,4,2,32]));
  env.set("_2716", builder["mul"](env.get("_2706"), env.get("_2715"), {"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_mul_cos_2351"}));
  env.set("_2717", builder.reshape(env.get("_2716"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_2706"), 2, {"axis":3,"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_split_partial_input0_2343"});
    env.set("_2707", tmp[0]);
    env.set("_2708", tmp[1]);
  }
  env.set("_2709", builder.concat([env.get("_2708"), env.get("_2707")], 3, {"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_concat_partial_input0_2344"}));
  env.set("Inserted_1251", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2335","maxValue":2047,"minValue":-2048}));
  env.set("_2696", builder["gather"](env.get("_621"), env.get("Inserted_1251"), {"axis":0,"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_gather_sin_2334"}));
  env.set("_2697", builder.reshape(env.get("_2696"), [1,sequence_length,1,1,32]));
  env.set("_2710", builder["mul"](env.get("_2709"), env.get("_2697"), {"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_mul_sin_2345"}));
  env.set("_2712", builder["mul"](env.get("_2710"), env.get("_2711"), {"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_mul_sign_2346"}));
  env.set("_2713", builder.reshape(env.get("_2712"), [1,sequence_length,4,64]));
  env.set("_2718", builder["add"](env.get("_2717"), env.get("_2713"), {"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_2353"}));
  env.set("_2719", builder.reshape(env.get("_2718"), [1,sequence_length,256]));
  env.set("_2720", builder.reshape(env.get("_2719"), [1,sequence_length,4,64]));
  env.set("present_19_key_38", builder["scatterND"](env.get("past_key_values_19_key_2721"), env.get("Inserted_1271"), env.get("_2720"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/present_key/ScatterND_2356"}));
  env.set("_2722", builder.reshape(env.get("present_19_key_38"), [1,4,1,past_sequence_length,64]));
  env.set("_2723", builder.expand(env.get("_2722"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/true_present_key/expand_2361"}));
  env.set("_2724", builder.reshape(env.get("_2723"), [1,32,past_sequence_length,64]));
  env.set("_2725", builder["transpose"](env.get("_2724"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/present_key/transpose_2363","permutation":[0,1,3,2]}));
  env.set("_2752", builder["matmul"](env.get("_2751"), env.get("_2725"), {"label":"/model/layers.19/attn/GroupQueryAttention_/Attention/qkv/matmul_1_2387"}));
  env.set("_2753", builder["mul"](env.get("_2752"), env.get("_681"), {"label":"/model/layers.19/attn/GroupQueryAttention_/Attention/qkv/div_2388"}));
  env.set("_2692", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_2328"}));
  env.set("_2693", builder.cumulativeSum(env.get("_2692"), 3, {"exclusive":true,"label":"/model/layers.19/attn/GroupQueryAttention_range_of_mask_shape_2329"}));
  env.set("_2689", builder["add"](env.get("_610"), env.get("_2681"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/attn_mask/add_2325"}));
  env.set("_2690", builder.expand(env.get("_2689"), [past_sequence_length,sequence_length], {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/expand_neq_right_2326"}));
  env.set("_2691", builder["transpose"](env.get("_2690"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/neq_right/transpose_2327","permutation":[1,0]}));
  env.set("Inserted_1249", builder["lesser"](env.get("_2693"), env.get("_2691"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_2330"}));
  env.set("_2694", builder.cast(env.get("Inserted_1249"), "uint8"));
  env.set("Inserted_1250", builder.cast(env.get("_2694"), "uint8"));
  env.set("_2695", builder["where"](env.get("Inserted_1250"), env.get("_618"), env.get("_619"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/attn_mask/where_2332"}));
  env.set("_2754", builder["add"](env.get("_2753"), env.get("_2695"), {"label":"/model/layers.19/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_2389"}));
  env.set("_2755", builder["softmax"](env.get("_2754"), 3));
  env.set("Inserted_1238", builder.cast(env.get("_2684"), "int64"));
  env.set("Inserted_1240", builder["max"](env.get("Inserted_1238"), env.get("Inserted_1239"), {"label":"Inserted_Max_2320"}));
  env.set("Inserted_1242", builder["min"](env.get("Inserted_1240"), env.get("Inserted_1241"), {"label":"Inserted_Min_2321"}));
  env.set("_120", builder.dequantizeLinear(env.get("_117"), env.get("_118"), env.get("_119"), {"axis":2,"blockSize":32,"label":"/model/layers.19/attn/v_proj/MatMul_Q4_dequantizeLinear_36"}));
  env.set("_121", builder.reshape(env.get("_120"), [256,2048]));
  env.set("_122", builder["transpose"](env.get("_121"), {"label":"/model/layers.19/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_38","permutation":[1,0]}));
  env.set("_2679", builder["matmul"](env.get("_2678"), env.get("_122"), {"label":"/model/layers.19/attn/v_proj/MatMul_Q4_matmul_2311"}));
  env.set("_2680", builder.reshape(env.get("_2679"), [1,sequence_length,4,64]));
  env.set("present_19_value_39", builder["scatterND"](env.get("past_key_values_19_value_2685"), env.get("Inserted_1242"), env.get("_2680"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/present_value/ScatterND_2318"}));
  env.set("_2686", builder.reshape(env.get("present_19_value_39"), [1,4,1,past_sequence_length,64]));
  env.set("_2687", builder.expand(env.get("_2686"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/true_present_value/expand_2323"}));
  env.set("_2688", builder.reshape(env.get("_2687"), [1,32,past_sequence_length,64]));
  env.set("_2756", builder["matmul"](env.get("_2755"), env.get("_2688"), {"label":"/model/layers.19/attn/GroupQueryAttention_/Attention/qkv/matmul_2_2391"}));
  env.set("_2757", builder["transpose"](env.get("_2756"), {"label":"/model/layers.19/attn/GroupQueryAttention_/Attention/qkv/transpose_2392","permutation":[0,2,1,3]}));
  env.set("_2758", builder.reshape(env.get("_2757"), [1,sequence_length,2048]));
  env.set("_114", builder.dequantizeLinear(env.get("_111"), env.get("_112"), env.get("_113"), {"axis":2,"blockSize":32,"label":"/model/layers.19/attn/o_proj/MatMul_Q4_dequantizeLinear_33"}));
  env.set("_115", builder.reshape(env.get("_114"), [2048,2048]));
  env.set("_116", builder["transpose"](env.get("_115"), {"label":"/model/layers.19/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_35","permutation":[1,0]}));
  env.set("_2759", builder["matmul"](env.get("_2758"), env.get("_116"), {"label":"/model/layers.19/attn/o_proj/MatMul_Q4_matmul_2394"}));
  env.set("_2760", builder["add"](env.get("_2671"), env.get("_2759"), {"label":"/model/layers.19/post_attention_layernorm/SkipLayerNorm_add_skip_2395"}));
  env.set("_2761", builder["pow"](env.get("_2760"), env.get("_582"), {"label":"/model/layers.19/post_attention_layernorm/SkipLayerNorm_pow_2396"}));
  env.set("_2762", builder["reduceMean"](env.get("_2761"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.19/post_attention_layernorm/SkipLayerNorm_reduceMean_2397"}));
  env.set("_2763", builder["add"](env.get("_2762"), env.get("_585"), {"label":"/model/layers.19/post_attention_layernorm/SkipLayerNorm_add_2398"}));
  env.set("_2764", builder["sqrt"](env.get("_2763"), {"label":"/model/layers.19/post_attention_layernorm/SkipLayerNorm_sqrt_2399"}));
  env.set("_2765", builder["div"](env.get("_2760"), env.get("_2764"), {"label":"/model/layers.19/post_attention_layernorm/SkipLayerNorm_div_2400"}));
  env.set("_2767", builder["mul"](env.get("_2766"), env.get("_2765"), {"label":"/model/layers.19/post_attention_layernorm/SkipLayerNorm_mul_2401"}));
  env.set("_2775", builder["matmul"](env.get("_2767"), env.get("_2774"), {"label":"/model/layers.19/mlp/gate_proj/MatMul_Q4_matmul_2406"}));
  env.set("_2776", builder["sigmoid"](env.get("_2775"), {"label":"/model/layers.19/mlp/act_fn/Sigmoid_2407"}));
  env.set("_2777", builder["mul"](env.get("_2775"), env.get("_2776"), {"label":"/model/layers.19/mlp/act_fn/Mul_2408"}));
  env.set("_108", builder.dequantizeLinear(env.get("_105"), env.get("_106"), env.get("_107"), {"axis":2,"blockSize":32,"label":"/model/layers.19/mlp/up_proj/MatMul_Q4_dequantizeLinear_30"}));
  env.set("_109", builder.reshape(env.get("_108"), [5632,2048]));
  env.set("_110", builder["transpose"](env.get("_109"), {"label":"/model/layers.19/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_32","permutation":[1,0]}));
  env.set("_2768", builder["matmul"](env.get("_2767"), env.get("_110"), {"label":"/model/layers.19/mlp/up_proj/MatMul_Q4_matmul_2402"}));
  env.set("_2778", builder["mul"](env.get("_2777"), env.get("_2768"), {"label":"/model/layers.19/mlp/Mul_2409"}));
  env.set("_102", builder.dequantizeLinear(env.get("_99"), env.get("_100"), env.get("_101"), {"axis":2,"blockSize":32,"label":"/model/layers.19/mlp/down_proj/MatMul_Q4_dequantizeLinear_27"}));
  env.set("_103", builder.reshape(env.get("_102"), [2048,5632]));
  env.set("_104", builder["transpose"](env.get("_103"), {"label":"/model/layers.19/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_29","permutation":[1,0]}));
  env.set("_2779", builder["matmul"](env.get("_2778"), env.get("_104"), {"label":"/model/layers.19/mlp/down_proj/MatMul_Q4_matmul_2410"}));
  env.set("_2780", builder["add"](env.get("_2760"), env.get("_2779"), {"label":"/model/layers.20/input_layernorm/SkipLayerNorm_add_skip_2411"}));
  env.set("_2781", builder["pow"](env.get("_2780"), env.get("_582"), {"label":"/model/layers.20/input_layernorm/SkipLayerNorm_pow_2412"}));
  env.set("_2782", builder["reduceMean"](env.get("_2781"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.20/input_layernorm/SkipLayerNorm_reduceMean_2413"}));
  env.set("_2783", builder["add"](env.get("_2782"), env.get("_585"), {"label":"/model/layers.20/input_layernorm/SkipLayerNorm_add_2414"}));
  env.set("_2784", builder["sqrt"](env.get("_2783"), {"label":"/model/layers.20/input_layernorm/SkipLayerNorm_sqrt_2415"}));
  env.set("_2785", builder["div"](env.get("_2780"), env.get("_2784"), {"label":"/model/layers.20/input_layernorm/SkipLayerNorm_div_2416"}));
  env.set("_2787", builder["mul"](env.get("_2786"), env.get("_2785"), {"label":"/model/layers.20/input_layernorm/SkipLayerNorm_mul_2417"}));
  env.set("_2843", builder["matmul"](env.get("_2787"), env.get("_2842"), {"label":"/model/layers.20/attn/q_proj/MatMul_Q4_matmul_2477"}));
  env.set("_2844", builder.reshape(env.get("_2843"), [1,sequence_length,32,64]));
  env.set("_2845", builder.reshape(env.get("_2844"), [1,sequence_length,32,2,32]));
  env.set("_2855", builder["mul"](env.get("_2845"), env.get("_2854"), {"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_mul_cos_2488"}));
  env.set("_2856", builder.reshape(env.get("_2855"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_2845"), 2, {"axis":3,"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_split_partial_input0_2480"});
    env.set("_2846", tmp[0]);
    env.set("_2847", tmp[1]);
  }
  env.set("_2848", builder.concat([env.get("_2847"), env.get("_2846")], 3, {"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_concat_partial_input0_2481"}));
  env.set("Inserted_1335", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2472","maxValue":2047,"minValue":-2048}));
  env.set("_2835", builder["gather"](env.get("_621"), env.get("Inserted_1335"), {"axis":0,"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_gather_sin_2471"}));
  env.set("_2836", builder.reshape(env.get("_2835"), [1,sequence_length,1,1,32]));
  env.set("_2849", builder["mul"](env.get("_2848"), env.get("_2836"), {"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_mul_sin_2482"}));
  env.set("_2851", builder["mul"](env.get("_2849"), env.get("_2850"), {"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_mul_sign_2483"}));
  env.set("_2852", builder.reshape(env.get("_2851"), [1,sequence_length,32,64]));
  env.set("_2857", builder["add"](env.get("_2856"), env.get("_2852"), {"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_2490"}));
  env.set("_2858", builder.reshape(env.get("_2857"), [1,sequence_length,2048]));
  env.set("_2859", builder.reshape(env.get("_2858"), [1,sequence_length,32,64]));
  env.set("_2860", builder["transpose"](env.get("_2859"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/query/transpose_2493","permutation":[0,2,1,3]}));
  env.set("Inserted_1296", builder.cast(env.get("_598"), "uint8"));
  env.set("_2790", builder["where"](env.get("Inserted_1296"), env.get("_599"), env.get("_597"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/scatter/where_2420"}));
  env.set("_2791", builder["add"](env.get("_601"), env.get("_2790"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/right_constant/add_2422"}));
  env.set("_2792", builder.concat([env.get("_603"), env.get("_2791")], 2, {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_2423"}));
  env.set("_2793", builder.reshape(env.get("_2792"), [1,sequence_length,4,3]));
  env.set("Inserted_1327", builder.cast(env.get("_2793"), "int64"));
  env.set("Inserted_1329", builder["max"](env.get("Inserted_1327"), env.get("Inserted_1328"), {"label":"Inserted_Max_2465"}));
  env.set("Inserted_1331", builder["min"](env.get("Inserted_1329"), env.get("Inserted_1330"), {"label":"Inserted_Min_2466"}));
  env.set("Inserted_1320", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2456","maxValue":2047,"minValue":-2048}));
  env.set("_2823", builder["gather"](env.get("_641"), env.get("Inserted_1320"), {"axis":0,"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_gather_cos_2455"}));
  env.set("_2824", builder.reshape(env.get("_2823"), [1,sequence_length,1,1,32]));
  env.set("_2810", builder.dequantizeLinear(env.get("_2807"), env.get("_2808"), env.get("_2809"), {"axis":2,"blockSize":32,"label":"/model/layers.20/attn/k_proj/MatMul_Q4_dequantizeLinear_2444"}));
  env.set("_2811", builder.reshape(env.get("_2810"), [256,2048]));
  env.set("_2812", builder["transpose"](env.get("_2811"), {"label":"/model/layers.20/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_2446","permutation":[1,0]}));
  env.set("_2813", builder["matmul"](env.get("_2787"), env.get("_2812"), {"label":"/model/layers.20/attn/k_proj/MatMul_Q4_matmul_2447"}));
  env.set("_2814", builder.reshape(env.get("_2813"), [1,sequence_length,4,64]));
  env.set("_2815", builder.reshape(env.get("_2814"), [1,sequence_length,4,2,32]));
  env.set("_2825", builder["mul"](env.get("_2815"), env.get("_2824"), {"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_mul_cos_2458"}));
  env.set("_2826", builder.reshape(env.get("_2825"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_2815"), 2, {"axis":3,"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_split_partial_input0_2450"});
    env.set("_2816", tmp[0]);
    env.set("_2817", tmp[1]);
  }
  env.set("_2818", builder.concat([env.get("_2817"), env.get("_2816")], 3, {"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_concat_partial_input0_2451"}));
  env.set("Inserted_1311", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2442","maxValue":2047,"minValue":-2048}));
  env.set("_2805", builder["gather"](env.get("_621"), env.get("Inserted_1311"), {"axis":0,"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_gather_sin_2441"}));
  env.set("_2806", builder.reshape(env.get("_2805"), [1,sequence_length,1,1,32]));
  env.set("_2819", builder["mul"](env.get("_2818"), env.get("_2806"), {"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_mul_sin_2452"}));
  env.set("_2821", builder["mul"](env.get("_2819"), env.get("_2820"), {"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_mul_sign_2453"}));
  env.set("_2822", builder.reshape(env.get("_2821"), [1,sequence_length,4,64]));
  env.set("_2827", builder["add"](env.get("_2826"), env.get("_2822"), {"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_2460"}));
  env.set("_2828", builder.reshape(env.get("_2827"), [1,sequence_length,256]));
  env.set("_2829", builder.reshape(env.get("_2828"), [1,sequence_length,4,64]));
  env.set("present_20_key_40", builder["scatterND"](env.get("past_key_values_20_key_2830"), env.get("Inserted_1331"), env.get("_2829"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/present_key/ScatterND_2463"}));
  env.set("_2831", builder.reshape(env.get("present_20_key_40"), [1,4,1,past_sequence_length,64]));
  env.set("_2832", builder.expand(env.get("_2831"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/true_present_key/expand_2468"}));
  env.set("_2833", builder.reshape(env.get("_2832"), [1,32,past_sequence_length,64]));
  env.set("_2834", builder["transpose"](env.get("_2833"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/present_key/transpose_2470","permutation":[0,1,3,2]}));
  env.set("_2861", builder["matmul"](env.get("_2860"), env.get("_2834"), {"label":"/model/layers.20/attn/GroupQueryAttention_/Attention/qkv/matmul_1_2494"}));
  env.set("_2862", builder["mul"](env.get("_2861"), env.get("_681"), {"label":"/model/layers.20/attn/GroupQueryAttention_/Attention/qkv/div_2495"}));
  env.set("_2801", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_2435"}));
  env.set("_2802", builder.cumulativeSum(env.get("_2801"), 3, {"exclusive":true,"label":"/model/layers.20/attn/GroupQueryAttention_range_of_mask_shape_2436"}));
  env.set("_2798", builder["add"](env.get("_610"), env.get("_2790"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/attn_mask/add_2432"}));
  env.set("_2799", builder.expand(env.get("_2798"), [past_sequence_length,sequence_length], {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/expand_neq_right_2433"}));
  env.set("_2800", builder["transpose"](env.get("_2799"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/neq_right/transpose_2434","permutation":[1,0]}));
  env.set("Inserted_1309", builder["lesser"](env.get("_2802"), env.get("_2800"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_2437"}));
  env.set("_2803", builder.cast(env.get("Inserted_1309"), "uint8"));
  env.set("Inserted_1310", builder.cast(env.get("_2803"), "uint8"));
  env.set("_2804", builder["where"](env.get("Inserted_1310"), env.get("_618"), env.get("_619"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/attn_mask/where_2439"}));
  env.set("_2863", builder["add"](env.get("_2862"), env.get("_2804"), {"label":"/model/layers.20/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_2496"}));
  env.set("_2864", builder["softmax"](env.get("_2863"), 3));
  env.set("Inserted_1298", builder.cast(env.get("_2793"), "int64"));
  env.set("Inserted_1300", builder["max"](env.get("Inserted_1298"), env.get("Inserted_1299"), {"label":"Inserted_Max_2427"}));
  env.set("Inserted_1302", builder["min"](env.get("Inserted_1300"), env.get("Inserted_1301"), {"label":"Inserted_Min_2428"}));
  env.set("_96", builder.dequantizeLinear(env.get("_93"), env.get("_94"), env.get("_95"), {"axis":2,"blockSize":32,"label":"/model/layers.20/attn/v_proj/MatMul_Q4_dequantizeLinear_24"}));
  env.set("_97", builder.reshape(env.get("_96"), [256,2048]));
  env.set("_98", builder["transpose"](env.get("_97"), {"label":"/model/layers.20/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_26","permutation":[1,0]}));
  env.set("_2788", builder["matmul"](env.get("_2787"), env.get("_98"), {"label":"/model/layers.20/attn/v_proj/MatMul_Q4_matmul_2418"}));
  env.set("_2789", builder.reshape(env.get("_2788"), [1,sequence_length,4,64]));
  env.set("present_20_value_41", builder["scatterND"](env.get("past_key_values_20_value_2794"), env.get("Inserted_1302"), env.get("_2789"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/present_value/ScatterND_2425"}));
  env.set("_2795", builder.reshape(env.get("present_20_value_41"), [1,4,1,past_sequence_length,64]));
  env.set("_2796", builder.expand(env.get("_2795"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/true_present_value/expand_2430"}));
  env.set("_2797", builder.reshape(env.get("_2796"), [1,32,past_sequence_length,64]));
  env.set("_2865", builder["matmul"](env.get("_2864"), env.get("_2797"), {"label":"/model/layers.20/attn/GroupQueryAttention_/Attention/qkv/matmul_2_2498"}));
  env.set("_2866", builder["transpose"](env.get("_2865"), {"label":"/model/layers.20/attn/GroupQueryAttention_/Attention/qkv/transpose_2499","permutation":[0,2,1,3]}));
  env.set("_2867", builder.reshape(env.get("_2866"), [1,sequence_length,2048]));
  env.set("_90", builder.dequantizeLinear(env.get("_87"), env.get("_88"), env.get("_89"), {"axis":2,"blockSize":32,"label":"/model/layers.20/attn/o_proj/MatMul_Q4_dequantizeLinear_21"}));
  env.set("_91", builder.reshape(env.get("_90"), [2048,2048]));
  env.set("_92", builder["transpose"](env.get("_91"), {"label":"/model/layers.20/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_23","permutation":[1,0]}));
  env.set("_2868", builder["matmul"](env.get("_2867"), env.get("_92"), {"label":"/model/layers.20/attn/o_proj/MatMul_Q4_matmul_2501"}));
  env.set("_2869", builder["add"](env.get("_2780"), env.get("_2868"), {"label":"/model/layers.20/post_attention_layernorm/SkipLayerNorm_add_skip_2502"}));
  env.set("_2870", builder["pow"](env.get("_2869"), env.get("_582"), {"label":"/model/layers.20/post_attention_layernorm/SkipLayerNorm_pow_2503"}));
  env.set("_2871", builder["reduceMean"](env.get("_2870"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.20/post_attention_layernorm/SkipLayerNorm_reduceMean_2504"}));
  env.set("_2872", builder["add"](env.get("_2871"), env.get("_585"), {"label":"/model/layers.20/post_attention_layernorm/SkipLayerNorm_add_2505"}));
  env.set("_2873", builder["sqrt"](env.get("_2872"), {"label":"/model/layers.20/post_attention_layernorm/SkipLayerNorm_sqrt_2506"}));
  env.set("_2874", builder["div"](env.get("_2869"), env.get("_2873"), {"label":"/model/layers.20/post_attention_layernorm/SkipLayerNorm_div_2507"}));
  env.set("_2876", builder["mul"](env.get("_2875"), env.get("_2874"), {"label":"/model/layers.20/post_attention_layernorm/SkipLayerNorm_mul_2508"}));
  env.set("_2884", builder["matmul"](env.get("_2876"), env.get("_2883"), {"label":"/model/layers.20/mlp/gate_proj/MatMul_Q4_matmul_2513"}));
  env.set("_2885", builder["sigmoid"](env.get("_2884"), {"label":"/model/layers.20/mlp/act_fn/Sigmoid_2514"}));
  env.set("_2886", builder["mul"](env.get("_2884"), env.get("_2885"), {"label":"/model/layers.20/mlp/act_fn/Mul_2515"}));
  env.set("_84", builder.dequantizeLinear(env.get("_81"), env.get("_82"), env.get("_83"), {"axis":2,"blockSize":32,"label":"/model/layers.20/mlp/up_proj/MatMul_Q4_dequantizeLinear_18"}));
  env.set("_85", builder.reshape(env.get("_84"), [5632,2048]));
  env.set("_86", builder["transpose"](env.get("_85"), {"label":"/model/layers.20/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_20","permutation":[1,0]}));
  env.set("_2877", builder["matmul"](env.get("_2876"), env.get("_86"), {"label":"/model/layers.20/mlp/up_proj/MatMul_Q4_matmul_2509"}));
  env.set("_2887", builder["mul"](env.get("_2886"), env.get("_2877"), {"label":"/model/layers.20/mlp/Mul_2516"}));
  env.set("_78", builder.dequantizeLinear(env.get("_75"), env.get("_76"), env.get("_77"), {"axis":2,"blockSize":32,"label":"/model/layers.20/mlp/down_proj/MatMul_Q4_dequantizeLinear_15"}));
  env.set("_79", builder.reshape(env.get("_78"), [2048,5632]));
  env.set("_80", builder["transpose"](env.get("_79"), {"label":"/model/layers.20/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_17","permutation":[1,0]}));
  env.set("_2888", builder["matmul"](env.get("_2887"), env.get("_80"), {"label":"/model/layers.20/mlp/down_proj/MatMul_Q4_matmul_2517"}));
  env.set("_2889", builder["add"](env.get("_2869"), env.get("_2888"), {"label":"/model/layers.21/input_layernorm/SkipLayerNorm_add_skip_2518"}));
  env.set("_2890", builder["pow"](env.get("_2889"), env.get("_582"), {"label":"/model/layers.21/input_layernorm/SkipLayerNorm_pow_2519"}));
  env.set("_2891", builder["reduceMean"](env.get("_2890"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.21/input_layernorm/SkipLayerNorm_reduceMean_2520"}));
  env.set("_2892", builder["add"](env.get("_2891"), env.get("_585"), {"label":"/model/layers.21/input_layernorm/SkipLayerNorm_add_2521"}));
  env.set("_2893", builder["sqrt"](env.get("_2892"), {"label":"/model/layers.21/input_layernorm/SkipLayerNorm_sqrt_2522"}));
  env.set("_2894", builder["div"](env.get("_2889"), env.get("_2893"), {"label":"/model/layers.21/input_layernorm/SkipLayerNorm_div_2523"}));
  env.set("_2896", builder["mul"](env.get("_2895"), env.get("_2894"), {"label":"/model/layers.21/input_layernorm/SkipLayerNorm_mul_2524"}));
  env.set("_2952", builder["matmul"](env.get("_2896"), env.get("_2951"), {"label":"/model/layers.21/attn/q_proj/MatMul_Q4_matmul_2584"}));
  env.set("_2953", builder.reshape(env.get("_2952"), [1,sequence_length,32,64]));
  env.set("_2954", builder.reshape(env.get("_2953"), [1,sequence_length,32,2,32]));
  env.set("_2964", builder["mul"](env.get("_2954"), env.get("_2963"), {"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_mul_cos_2595"}));
  env.set("_2965", builder.reshape(env.get("_2964"), [1,sequence_length,32,64]));
  {
    const tmp = builder.split(env.get("_2954"), 2, {"axis":3,"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_split_partial_input0_2587"});
    env.set("_2955", tmp[0]);
    env.set("_2956", tmp[1]);
  }
  env.set("_2957", builder.concat([env.get("_2956"), env.get("_2955")], 3, {"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_concat_partial_input0_2588"}));
  env.set("Inserted_1395", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2579","maxValue":2047,"minValue":-2048}));
  env.set("_2944", builder["gather"](env.get("_621"), env.get("Inserted_1395"), {"axis":0,"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_gather_sin_2578"}));
  env.set("_2945", builder.reshape(env.get("_2944"), [1,sequence_length,1,1,32]));
  env.set("_2958", builder["mul"](env.get("_2957"), env.get("_2945"), {"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_mul_sin_2589"}));
  env.set("_2960", builder["mul"](env.get("_2958"), env.get("_2959"), {"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_mul_sign_2590"}));
  env.set("_2961", builder.reshape(env.get("_2960"), [1,sequence_length,32,64]));
  env.set("_2966", builder["add"](env.get("_2965"), env.get("_2961"), {"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_2597"}));
  env.set("_2967", builder.reshape(env.get("_2966"), [1,sequence_length,2048]));
  env.set("_2968", builder.reshape(env.get("_2967"), [1,sequence_length,32,64]));
  env.set("_2969", builder["transpose"](env.get("_2968"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/query/transpose_2600","permutation":[0,2,1,3]}));
  env.set("Inserted_1356", builder.cast(env.get("_598"), "uint8"));
  env.set("_2899", builder["where"](env.get("Inserted_1356"), env.get("_599"), env.get("_597"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/scatter/where_2527"}));
  env.set("_2900", builder["add"](env.get("_601"), env.get("_2899"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/right_constant/add_2529"}));
  env.set("_2901", builder.concat([env.get("_603"), env.get("_2900")], 2, {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_2530"}));
  env.set("_2902", builder.reshape(env.get("_2901"), [1,sequence_length,4,3]));
  env.set("Inserted_1387", builder.cast(env.get("_2902"), "int64"));
  env.set("Inserted_1389", builder["max"](env.get("Inserted_1387"), env.get("Inserted_1388"), {"label":"Inserted_Max_2572"}));
  env.set("Inserted_1391", builder["min"](env.get("Inserted_1389"), env.get("Inserted_1390"), {"label":"Inserted_Min_2573"}));
  env.set("Inserted_1380", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2563","maxValue":2047,"minValue":-2048}));
  env.set("_2932", builder["gather"](env.get("_641"), env.get("Inserted_1380"), {"axis":0,"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_gather_cos_2562"}));
  env.set("_2933", builder.reshape(env.get("_2932"), [1,sequence_length,1,1,32]));
  env.set("_2919", builder.dequantizeLinear(env.get("_2916"), env.get("_2917"), env.get("_2918"), {"axis":2,"blockSize":32,"label":"/model/layers.21/attn/k_proj/MatMul_Q4_dequantizeLinear_2551"}));
  env.set("_2920", builder.reshape(env.get("_2919"), [256,2048]));
  env.set("_2921", builder["transpose"](env.get("_2920"), {"label":"/model/layers.21/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_2553","permutation":[1,0]}));
  env.set("_2922", builder["matmul"](env.get("_2896"), env.get("_2921"), {"label":"/model/layers.21/attn/k_proj/MatMul_Q4_matmul_2554"}));
  env.set("_2923", builder.reshape(env.get("_2922"), [1,sequence_length,4,64]));
  env.set("_2924", builder.reshape(env.get("_2923"), [1,sequence_length,4,2,32]));
  env.set("_2934", builder["mul"](env.get("_2924"), env.get("_2933"), {"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_mul_cos_2565"}));
  env.set("_2935", builder.reshape(env.get("_2934"), [1,sequence_length,4,64]));
  {
    const tmp = builder.split(env.get("_2924"), 2, {"axis":3,"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_split_partial_input0_2557"});
    env.set("_2925", tmp[0]);
    env.set("_2926", tmp[1]);
  }
  env.set("_2927", builder.concat([env.get("_2926"), env.get("_2925")], 3, {"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_concat_partial_input0_2558"}));
  env.set("Inserted_1371", builder["clamp"](env.get("position_ids_622"), {"label":"Inserted_Clip_2549","maxValue":2047,"minValue":-2048}));
  env.set("_2914", builder["gather"](env.get("_621"), env.get("Inserted_1371"), {"axis":0,"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_gather_sin_2548"}));
  env.set("_2915", builder.reshape(env.get("_2914"), [1,sequence_length,1,1,32]));
  env.set("_2928", builder["mul"](env.get("_2927"), env.get("_2915"), {"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_mul_sin_2559"}));
  env.set("_2930", builder["mul"](env.get("_2928"), env.get("_2929"), {"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_mul_sign_2560"}));
  env.set("_2931", builder.reshape(env.get("_2930"), [1,sequence_length,4,64]));
  env.set("_2936", builder["add"](env.get("_2935"), env.get("_2931"), {"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_2567"}));
  env.set("_2937", builder.reshape(env.get("_2936"), [1,sequence_length,256]));
  env.set("_2938", builder.reshape(env.get("_2937"), [1,sequence_length,4,64]));
  env.set("present_21_key_42", builder["scatterND"](env.get("past_key_values_21_key_2939"), env.get("Inserted_1391"), env.get("_2938"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/present_key/ScatterND_2570"}));
  env.set("_2940", builder.reshape(env.get("present_21_key_42"), [1,4,1,past_sequence_length,64]));
  env.set("_2941", builder.expand(env.get("_2940"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/true_present_key/expand_2575"}));
  env.set("_2942", builder.reshape(env.get("_2941"), [1,32,past_sequence_length,64]));
  env.set("_2943", builder["transpose"](env.get("_2942"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/present_key/transpose_2577","permutation":[0,1,3,2]}));
  env.set("_2970", builder["matmul"](env.get("_2969"), env.get("_2943"), {"label":"/model/layers.21/attn/GroupQueryAttention_/Attention/qkv/matmul_1_2601"}));
  env.set("_2971", builder["mul"](env.get("_2970"), env.get("_681"), {"label":"/model/layers.21/attn/GroupQueryAttention_/Attention/qkv/div_2602"}));
  env.set("_2910", builder.expand(env.get("_614"), [1,32,sequence_length,past_sequence_length], {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_2542"}));
  env.set("_2911", builder.cumulativeSum(env.get("_2910"), 3, {"exclusive":true,"label":"/model/layers.21/attn/GroupQueryAttention_range_of_mask_shape_2543"}));
  env.set("_2907", builder["add"](env.get("_610"), env.get("_2899"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/attn_mask/add_2539"}));
  env.set("_2908", builder.expand(env.get("_2907"), [past_sequence_length,sequence_length], {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/expand_neq_right_2540"}));
  env.set("_2909", builder["transpose"](env.get("_2908"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/neq_right/transpose_2541","permutation":[1,0]}));
  env.set("Inserted_1369", builder["lesser"](env.get("_2911"), env.get("_2909"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_2544"}));
  env.set("_2912", builder.cast(env.get("Inserted_1369"), "uint8"));
  env.set("Inserted_1370", builder.cast(env.get("_2912"), "uint8"));
  env.set("_2913", builder["where"](env.get("Inserted_1370"), env.get("_618"), env.get("_619"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/attn_mask/where_2546"}));
  env.set("_2972", builder["add"](env.get("_2971"), env.get("_2913"), {"label":"/model/layers.21/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_2603"}));
  env.set("_2973", builder["softmax"](env.get("_2972"), 3));
  env.set("Inserted_1358", builder.cast(env.get("_2902"), "int64"));
  env.set("Inserted_1360", builder["max"](env.get("Inserted_1358"), env.get("Inserted_1359"), {"label":"Inserted_Max_2534"}));
  env.set("Inserted_1362", builder["min"](env.get("Inserted_1360"), env.get("Inserted_1361"), {"label":"Inserted_Min_2535"}));
  env.set("_72", builder.dequantizeLinear(env.get("_69"), env.get("_70"), env.get("_71"), {"axis":2,"blockSize":32,"label":"/model/layers.21/attn/v_proj/MatMul_Q4_dequantizeLinear_12"}));
  env.set("_73", builder.reshape(env.get("_72"), [256,2048]));
  env.set("_74", builder["transpose"](env.get("_73"), {"label":"/model/layers.21/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_14","permutation":[1,0]}));
  env.set("_2897", builder["matmul"](env.get("_2896"), env.get("_74"), {"label":"/model/layers.21/attn/v_proj/MatMul_Q4_matmul_2525"}));
  env.set("_2898", builder.reshape(env.get("_2897"), [1,sequence_length,4,64]));
  env.set("present_21_value_43", builder["scatterND"](env.get("past_key_values_21_value_2903"), env.get("Inserted_1362"), env.get("_2898"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/present_value/ScatterND_2532"}));
  env.set("_2904", builder.reshape(env.get("present_21_value_43"), [1,4,1,past_sequence_length,64]));
  env.set("_2905", builder.expand(env.get("_2904"), [1,4,8,past_sequence_length,64], {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/true_present_value/expand_2537"}));
  env.set("_2906", builder.reshape(env.get("_2905"), [1,32,past_sequence_length,64]));
  env.set("_2974", builder["matmul"](env.get("_2973"), env.get("_2906"), {"label":"/model/layers.21/attn/GroupQueryAttention_/Attention/qkv/matmul_2_2605"}));
  env.set("_2975", builder["transpose"](env.get("_2974"), {"label":"/model/layers.21/attn/GroupQueryAttention_/Attention/qkv/transpose_2606","permutation":[0,2,1,3]}));
  env.set("_2976", builder.reshape(env.get("_2975"), [1,sequence_length,2048]));
  env.set("_66", builder.dequantizeLinear(env.get("_63"), env.get("_64"), env.get("_65"), {"axis":2,"blockSize":32,"label":"/model/layers.21/attn/o_proj/MatMul_Q4_dequantizeLinear_9"}));
  env.set("_67", builder.reshape(env.get("_66"), [2048,2048]));
  env.set("_68", builder["transpose"](env.get("_67"), {"label":"/model/layers.21/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_11","permutation":[1,0]}));
  env.set("_2977", builder["matmul"](env.get("_2976"), env.get("_68"), {"label":"/model/layers.21/attn/o_proj/MatMul_Q4_matmul_2608"}));
  env.set("_2978", builder["add"](env.get("_2889"), env.get("_2977"), {"label":"/model/layers.21/post_attention_layernorm/SkipLayerNorm_add_skip_2609"}));
  env.set("_2979", builder["pow"](env.get("_2978"), env.get("_582"), {"label":"/model/layers.21/post_attention_layernorm/SkipLayerNorm_pow_2610"}));
  env.set("_2980", builder["reduceMean"](env.get("_2979"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.21/post_attention_layernorm/SkipLayerNorm_reduceMean_2611"}));
  env.set("_2981", builder["add"](env.get("_2980"), env.get("_585"), {"label":"/model/layers.21/post_attention_layernorm/SkipLayerNorm_add_2612"}));
  env.set("_2982", builder["sqrt"](env.get("_2981"), {"label":"/model/layers.21/post_attention_layernorm/SkipLayerNorm_sqrt_2613"}));
  env.set("_2983", builder["div"](env.get("_2978"), env.get("_2982"), {"label":"/model/layers.21/post_attention_layernorm/SkipLayerNorm_div_2614"}));
  env.set("_2985", builder["mul"](env.get("_2984"), env.get("_2983"), {"label":"/model/layers.21/post_attention_layernorm/SkipLayerNorm_mul_2615"}));
  env.set("_2993", builder["matmul"](env.get("_2985"), env.get("_2992"), {"label":"/model/layers.21/mlp/gate_proj/MatMul_Q4_matmul_2620"}));
  env.set("_2994", builder["sigmoid"](env.get("_2993"), {"label":"/model/layers.21/mlp/act_fn/Sigmoid_2621"}));
  env.set("_2995", builder["mul"](env.get("_2993"), env.get("_2994"), {"label":"/model/layers.21/mlp/act_fn/Mul_2622"}));
  env.set("_60", builder.dequantizeLinear(env.get("_57"), env.get("_58"), env.get("_59"), {"axis":2,"blockSize":32,"label":"/model/layers.21/mlp/up_proj/MatMul_Q4_dequantizeLinear_6"}));
  env.set("_61", builder.reshape(env.get("_60"), [5632,2048]));
  env.set("_62", builder["transpose"](env.get("_61"), {"label":"/model/layers.21/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_8","permutation":[1,0]}));
  env.set("_2986", builder["matmul"](env.get("_2985"), env.get("_62"), {"label":"/model/layers.21/mlp/up_proj/MatMul_Q4_matmul_2616"}));
  env.set("_2996", builder["mul"](env.get("_2995"), env.get("_2986"), {"label":"/model/layers.21/mlp/Mul_2623"}));
  env.set("_54", builder.dequantizeLinear(env.get("_51"), env.get("_52"), env.get("_53"), {"axis":2,"blockSize":32,"label":"/model/layers.21/mlp/down_proj/MatMul_Q4_dequantizeLinear_3"}));
  env.set("_55", builder.reshape(env.get("_54"), [2048,5632]));
  env.set("_56", builder["transpose"](env.get("_55"), {"label":"/model/layers.21/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_5","permutation":[1,0]}));
  env.set("_2997", builder["matmul"](env.get("_2996"), env.get("_56"), {"label":"/model/layers.21/mlp/down_proj/MatMul_Q4_matmul_2624"}));
  env.set("_2998", builder["add"](env.get("_2978"), env.get("_2997"), {"label":"/model/layers.22/final_norm_layernorm/SkipLayerNorm_add_skip_2625"}));
  env.set("_2999", builder["pow"](env.get("_2998"), env.get("_582"), {"label":"/model/layers.22/final_norm_layernorm/SkipLayerNorm_pow_2626"}));
  env.set("_3000", builder["reduceMean"](env.get("_2999"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.22/final_norm_layernorm/SkipLayerNorm_reduceMean_2627"}));
  env.set("_3001", builder["add"](env.get("_3000"), env.get("_585"), {"label":"/model/layers.22/final_norm_layernorm/SkipLayerNorm_add_2628"}));
  env.set("_3002", builder["sqrt"](env.get("_3001"), {"label":"/model/layers.22/final_norm_layernorm/SkipLayerNorm_sqrt_2629"}));
  env.set("_3003", builder["div"](env.get("_2998"), env.get("_3002"), {"label":"/model/layers.22/final_norm_layernorm/SkipLayerNorm_div_2630"}));
  env.set("_3005", builder["mul"](env.get("_3004"), env.get("_3003"), {"label":"/model/layers.22/final_norm_layernorm/SkipLayerNorm_mul_2631"}));
  env.set("_48", builder.dequantizeLinear(env.get("_45"), env.get("_46"), env.get("_47"), {"axis":2,"blockSize":32,"label":"/lm_head/MatMul_Q4_dequantizeLinear_0"}));
  env.set("_49", builder.reshape(env.get("_48"), [32000,2048]));
  env.set("_50", builder["transpose"](env.get("_49"), {"label":"/lm_head/MatMul_Q4_transpose_dequantizeLinear_2","permutation":[1,0]}));
  env.set("logits_44", builder["matmul"](env.get("_3005"), env.get("_50"), {"label":"/lm_head/MatMul_Q4_matmul_2632"}));

  const outputs = {};
  outputs["logits_44"] = env.get("logits_44");
  outputs["present_0_key_0"] = env.get("present_0_key_0");
  outputs["present_0_value_1"] = env.get("present_0_value_1");
  outputs["present_10_key_20"] = env.get("present_10_key_20");
  outputs["present_10_value_21"] = env.get("present_10_value_21");
  outputs["present_11_key_22"] = env.get("present_11_key_22");
  outputs["present_11_value_23"] = env.get("present_11_value_23");
  outputs["present_12_key_24"] = env.get("present_12_key_24");
  outputs["present_12_value_25"] = env.get("present_12_value_25");
  outputs["present_13_key_26"] = env.get("present_13_key_26");
  outputs["present_13_value_27"] = env.get("present_13_value_27");
  outputs["present_14_key_28"] = env.get("present_14_key_28");
  outputs["present_14_value_29"] = env.get("present_14_value_29");
  outputs["present_15_key_30"] = env.get("present_15_key_30");
  outputs["present_15_value_31"] = env.get("present_15_value_31");
  outputs["present_16_key_32"] = env.get("present_16_key_32");
  outputs["present_16_value_33"] = env.get("present_16_value_33");
  outputs["present_17_key_34"] = env.get("present_17_key_34");
  outputs["present_17_value_35"] = env.get("present_17_value_35");
  outputs["present_18_key_36"] = env.get("present_18_key_36");
  outputs["present_18_value_37"] = env.get("present_18_value_37");
  outputs["present_19_key_38"] = env.get("present_19_key_38");
  outputs["present_19_value_39"] = env.get("present_19_value_39");
  outputs["present_1_key_2"] = env.get("present_1_key_2");
  outputs["present_1_value_3"] = env.get("present_1_value_3");
  outputs["present_20_key_40"] = env.get("present_20_key_40");
  outputs["present_20_value_41"] = env.get("present_20_value_41");
  outputs["present_21_key_42"] = env.get("present_21_key_42");
  outputs["present_21_value_43"] = env.get("present_21_value_43");
  outputs["present_2_key_4"] = env.get("present_2_key_4");
  outputs["present_2_value_5"] = env.get("present_2_value_5");
  outputs["present_3_key_6"] = env.get("present_3_key_6");
  outputs["present_3_value_7"] = env.get("present_3_value_7");
  outputs["present_4_key_8"] = env.get("present_4_key_8");
  outputs["present_4_value_9"] = env.get("present_4_value_9");
  outputs["present_5_key_10"] = env.get("present_5_key_10");
  outputs["present_5_value_11"] = env.get("present_5_value_11");
  outputs["present_6_key_12"] = env.get("present_6_key_12");
  outputs["present_6_value_13"] = env.get("present_6_value_13");
  outputs["present_7_key_14"] = env.get("present_7_key_14");
  outputs["present_7_value_15"] = env.get("present_7_value_15");
  outputs["present_8_key_16"] = env.get("present_8_key_16");
  outputs["present_8_value_17"] = env.get("present_8_value_17");
  outputs["present_9_key_18"] = env.get("present_9_key_18");
  outputs["present_9_value_19"] = env.get("present_9_value_19");
  return await builder.build(outputs);
}
