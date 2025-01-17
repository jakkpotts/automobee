import { Fragment } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { Command } from 'cmdk';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

export function CommandPalette({ isOpen, onClose }: CommandPaletteProps) {
  return (
    <Transition.Root show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-gray-500 bg-opacity-25 transition-opacity" />
        </Transition.Child>

        <div className="fixed inset-0 z-10 overflow-y-auto p-4 sm:p-6 md:p-20">
          <Transition.Child
            as={Fragment}
            enter="ease-out duration-300"
            enterFrom="opacity-0 scale-95"
            enterTo="opacity-100 scale-100"
            leave="ease-in duration-200"
            leaveFrom="opacity-100 scale-100"
            leaveTo="opacity-0 scale-95"
          >
            <Dialog.Panel className="mx-auto max-w-2xl transform divide-y divide-gray-100 overflow-hidden rounded-xl bg-white shadow-2xl ring-1 ring-black ring-opacity-5 transition-all">
              <Command className="relative">
                <div className="flex items-center border-b border-gray-100 px-4">
                  <Command.Input
                    className="flex h-12 w-full border-0 bg-transparent text-sm text-gray-800 placeholder:text-gray-400 focus:ring-0"
                    placeholder="Search commands..."
                  />
                </div>
                <Command.List className="max-h-96 overflow-y-auto py-4 px-2">
                  <Command.Group heading="Cameras">
                    <Command.Item className="group px-4 py-2 rounded-lg aria-selected:bg-blue-50 hover:bg-gray-50">
                      View All Cameras
                    </Command.Item>
                    <Command.Item className="group px-4 py-2 rounded-lg aria-selected:bg-blue-50 hover:bg-gray-50">
                      Add New Camera
                    </Command.Item>
                  </Command.Group>
                  <Command.Group heading="Views">
                    <Command.Item className="group px-4 py-2 rounded-lg aria-selected:bg-blue-50 hover:bg-gray-50">
                      Switch to Map View
                    </Command.Item>
                    <Command.Item className="group px-4 py-2 rounded-lg aria-selected:bg-blue-50 hover:bg-gray-50">
                      Switch to Split View
                    </Command.Item>
                  </Command.Group>
                  <Command.Group heading="Settings">
                    <Command.Item className="group px-4 py-2 rounded-lg aria-selected:bg-blue-50 hover:bg-gray-50">
                      Open Settings
                    </Command.Item>
                    <Command.Item className="group px-4 py-2 rounded-lg aria-selected:bg-blue-50 hover:bg-gray-50">
                      Configure Alerts
                    </Command.Item>
                  </Command.Group>
                </Command.List>
              </Command>
            </Dialog.Panel>
          </Transition.Child>
        </div>
      </Dialog>
    </Transition.Root>
  );
} 