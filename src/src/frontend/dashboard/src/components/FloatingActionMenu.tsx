import { Fragment, useState } from 'react';
import { Menu, Transition } from '@headlessui/react';
import {
  ViewColumnsIcon,
  Squares2X2Icon,
  MapIcon,
  CameraIcon,
  ArrowsPointingOutIcon,
} from '@heroicons/react/24/outline';

interface FloatingActionMenuProps {
  onLayoutChange: (layout: string) => void;
}

export function FloatingActionMenu({ onLayoutChange }: FloatingActionMenuProps) {
  const [isOpen, setIsOpen] = useState(false);

  const menuItems = [
    {
      name: 'Default View',
      icon: Squares2X2Icon,
      onClick: () => onLayoutChange('default'),
    },
    {
      name: 'Split View',
      icon: ViewColumnsIcon,
      onClick: () => onLayoutChange('split'),
    },
    {
      name: 'Map Only',
      icon: MapIcon,
      onClick: () => onLayoutChange('fullscreen'),
    },
    {
      name: 'Camera Grid',
      icon: CameraIcon,
      onClick: () => onLayoutChange('grid'),
    },
  ];

  return (
    <div className="fixed bottom-4 left-4">
      <Menu as="div" className="relative inline-block text-left">
        <Menu.Button
          className="inline-flex items-center p-3 rounded-full shadow-lg bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          onClick={() => setIsOpen(!isOpen)}
        >
          <ArrowsPointingOutIcon className="h-6 w-6 text-gray-600" />
        </Menu.Button>

        <Transition
          as={Fragment}
          enter="transition ease-out duration-100"
          enterFrom="transform opacity-0 scale-95"
          enterTo="transform opacity-100 scale-100"
          leave="transition ease-in duration-75"
          leaveFrom="transform opacity-100 scale-100"
          leaveTo="transform opacity-0 scale-95"
        >
          <Menu.Items className="absolute bottom-full mb-2 left-0 w-56 rounded-lg bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
            <div className="py-1">
              {menuItems.map((item) => (
                <Menu.Item key={item.name}>
                  {({ active }) => (
                    <button
                      className={`${
                        active ? 'bg-gray-100 text-gray-900' : 'text-gray-700'
                      } group flex items-center w-full px-4 py-2 text-sm`}
                      onClick={() => {
                        item.onClick();
                        setIsOpen(false);
                      }}
                    >
                      <item.icon
                        className="mr-3 h-5 w-5 text-gray-400 group-hover:text-gray-500"
                        aria-hidden="true"
                      />
                      {item.name}
                    </button>
                  )}
                </Menu.Item>
              ))}
            </div>
          </Menu.Items>
        </Transition>
      </Menu>
    </div>
  );
} 