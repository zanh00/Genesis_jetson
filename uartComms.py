import serial
import struct
import time

class JetsonNanoToSTM:
    def __init__(self, baud_rate=115200):
        """Initialize the UART interface."""
        try:
            self.ser = serial.Serial('/dev/ttyTHS1', baudrate=baud_rate, timeout=1)
        except serial.SerialException as e:
            print(f"Failed to open serial port: {e}")
            raise

    def __float_to_ascii_hex(self, value):
        """Convert a 32-bit float to an 8-byte ASCII-encoded hex string."""
        # Convert float to 4 bytes using IEEE 754 format
        float_bytes = struct.pack('>f', value)
        # Convert each byte to two ASCII nibbles
        ascii_hex = ''.join(f"{byte:02X}" for byte in float_bytes)
        return ascii_hex

    def __id_to_ascii_hex(self, data_id):
        """Convert a 1-byte data ID to a 2-byte ASCII-encoded hex string."""
        if not (0 <= data_id <= 255):
            raise ValueError("Data ID must be in range 0-255.")
        return f"{data_id:02X}"

    def __create_message(self, data_id, data_value):
        """Create a message in the specified format."""

        # Encode ID and data
        encoded_id = self.__id_to_ascii_hex(data_id)
        encoded_data = self.__float_to_ascii_hex(data_value)
        # Construct the full message as a list of bytes
        start_byte = [ord('Z')]
        message =  [ord(c) for c in encoded_id + encoded_data]
        return start_byte, message

    def send_message(self, data_ids : list, data_values : list):
        """Encode and send a message via UART."""
        if len(data_ids) != len(data_values):
            raise ValueError("Data IDs and values must have the same length.")
        # Iterate over each data ID and value
        for data_id, data_value in zip(data_ids, data_values):  
            start_byte, message = self.__create_message(data_id, data_value)
            self.ser.write(start_byte)
            time.sleep(0.001)
            self.ser.write(message)

    def close(self):
        """Close the UART interface."""
        self.ser.close()

# Example usage
if __name__ == "__main__":
    # Initialize the UART handler
    uart_handler = JetsonNanoToSTM(baud_rate=115200)
    
    try:
        # Example data
        data_id = 129
        data_value = 123.456

        # Send a message
        uart_handler.send_message(data_id, data_value)
        print("Message sent successfully.")

    finally:
        # Clean up
        uart_handler.close()
